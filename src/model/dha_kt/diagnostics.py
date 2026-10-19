"""Read-only DHA candidate, posterior, calibration, sparsity and gradient diagnostics."""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from src.model.registry import build_model, checkpoint_model_name
from src.model.newq_v2.diagnostics import distribution, prediction_metrics
from src.training.trainer import load_checkpoint, make_loader, move_batch
from src.utils.config import load_config, project_path
from src.utils.files import sha256
from src.utils.reproducibility import seed_everything, select_device
from .candidates import generate_unique, unique_conditions
from src.training.dha_kt_stages import apply_stage


def calibration(logits, answers, bins=10):
    logits, answers = logits.detach().double().cpu(), answers.detach().double().cpu()
    probabilities = logits.sigmoid()
    indexes = (probabilities*bins).long().clamp(max=bins-1)
    ece, table = 0., []
    for index in range(bins):
        selected = indexes == index
        if selected.any():
            confidence, frequency = float(probabilities[selected].mean()), float(answers[selected].mean())
            ece += abs(confidence-frequency)*float(selected.double().mean())
            table.append(dict(bin=index, count=int(selected.sum()), mean_probability=confidence, correct_rate=frequency))
    return dict(nll=float(F.binary_cross_entropy_with_logits(logits,answers)), brier=float((probabilities-answers).square().mean()),
                ece=ece, bins=table)


def gradient_probe(model, batch, seed):
    groups = dict(item=model.item_encoder, diffusion=model.diffusion, adapter=model.adapter,
                  retriever=model.retriever, predictor=model.predictor)
    if hasattr(model,"profile_to_residual"):
        groups["profile_to_residual"] = model.profile_to_residual
    for name in ["attention_code","key_projection","residual_code","residual_projection","fusion_projection"]:
        if hasattr(model.item_encoder,name):
            groups[name] = getattr(model.item_encoder,name)
    model.train()
    model.set_data_context("train")
    apply_stage(model,model.training_stage)
    result={}
    for loss in ["kt_loss","pilot_loss","diffusion_loss"]:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        value=model(batch)[loss]
        if value is None or not value.requires_grad:
            result[loss]=None
            continue
        value.backward()
        result[loss]={}
        for name,module in groups.items():
            gradients=[p.grad.detach().double() for p in module.parameters() if p.grad is not None]
            norm=math.sqrt(sum(float(g.square().sum()) for g in gradients)) if gradients else None
            result[loss][name]=dict(has_gradient=bool(gradients),norm=norm,
                                    nonfinite_count=sum(int((~torch.isfinite(g)).sum()) for g in gradients))
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


def diagnose_checkpoint(config, checkpoint_path, split="valid", max_batches=2):
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive.")
    config=copy.deepcopy(config)
    control=config["train"]
    seed_everything(control["seed"],control["num_threads"])
    device=select_device(control["device"])
    control["device"]=str(device)
    checkpoint=load_checkpoint(checkpoint_path,device)
    if checkpoint_model_name(checkpoint)!="dha_kt":
        raise ValueError("DHA diagnostics require a dha_kt checkpoint.")
    processed=project_path(config["dataset"]["paths"]["processed"])
    if sha256(processed/"metadata.json")!=checkpoint["metadata_sha256"]:
        raise ValueError("Checkpoint and data metadata differ.")
    for field in ["max_windows","max_seq_len"]:
        if field not in control and field in checkpoint["config"]["train"]:
            control[field]=checkpoint["config"]["train"][field]
    model=build_model("dha_kt",checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"],strict=True)
    model.eval()
    model.set_data_context(split)
    metadata=model.experiment_metadata()
    generator=torch.Generator(device=device).manual_seed(control["eval_seed"])
    logits, labels, pilot_logits, problems=[],[],[],[]
    values={key:[] for key in ["candidate_norm","candidate_probability_variance","responsibility_entropy","responsibility_prior_l1",
                                "residual_semantic_ratio","correction_semantic_ratio"]}
    sampling_seconds,batches=0.,0
    with torch.no_grad():
        for batch in make_loader(config,split):
            if max_batches is not None and batches>=max_batches:
                break
            batch=move_batch(batch,device)
            output=model(batch,generator=generator,return_details=True)
            details=output["details"]
            valid,mask=batch["valid_mask"],batch["valid_mask"] & batch["prediction_mask"]
            logits.append(output["logits"][mask].cpu())
            labels.append(batch["answers"][mask].cpu())
            from .evidence import mixture_logit
            pilot_logits.append(mixture_logit(details["pilot_logits"],model.log_prior)[mask].cpu())
            problems.append(batch["problem_ids"][mask].cpu())
            values["candidate_norm"].append(details["candidates"].norm(dim=-1)[valid].cpu().flatten())
            values["candidate_probability_variance"].append(details["final_logits"].sigmoid().var(-1,unbiased=False)[mask].cpu())
            weights=details["responsibilities"]
            values["responsibility_entropy"].append((-(weights*weights.clamp_min(1e-30).log()).sum(-1))[valid].cpu())
            values["responsibility_prior_l1"].append((weights-model.log_prior.exp()).abs().sum(-1)[valid].cpu())
            parts=model.item_encoder(batch)
            keys,unique,_,_=unique_conditions(model,parts,batch)
            denominator=unique["semantic"].norm(dim=-1).clamp_min(1e-8)
            values["residual_semantic_ratio"].append((unique["residual"].norm(dim=-1)/denominator).cpu())
            if model.candidate_backend=="evidence_profile" and model.training_stage=="joint":
                _,correction=generate_unique(model,keys,unique,generator,return_correction=True)
                values["correction_semantic_ratio"].append((correction.norm(dim=-1)/denominator[:,None]).cpu().flatten())
            sampling_seconds+=float(output["sampling_seconds"])
            batches+=1
    if not logits:
        raise ValueError("No prediction batches in diagnostic scope.")
    logits,labels,pilot_logits,problems=map(torch.cat,[logits,labels,pilot_logits,problems])
    sparse={}
    if model.candidate_backend=="evidence_profile":
        bank=model.reference_bank
        counts=bank.dense_counts[bank.manifest["full_train_view"],0].sum(-1).cpu()[problems]
        for name,chosen in [("0",counts==0),("1-4",(counts>0)&(counts<5)),("5-19",(counts>=5)&(counts<20)),("20+",counts>=20)]:
            sparse[name]=dict(predictions=int(chosen.sum()),metrics=prediction_metrics(logits[chosen],labels[chosen]) if chosen.any() else None)
    gradient=gradient_probe(model,move_batch(next(iter(make_loader(config,"train"))),device),control["seed"])
    return dict(**metadata,model_name="dha_kt",checkpoint=str(project_path(checkpoint_path)),checkpoint_epoch=checkpoint["epoch"],
        device=str(device),split=split,batches=batches,predictions=prediction_metrics(logits,labels),
        calibration=calibration(logits,labels),pilot_calibration=calibration(pilot_logits,labels),
        statistics={key:distribution(torch.cat(v) if v else torch.empty(0)) for key,v in values.items()},
        reference_count_groups=sparse,gradient_probe=gradient,sampling_seconds=sampling_seconds,
        timing_scope="synchronized candidate building in selected evaluation batches; excludes reference building and extra diagnostic probes")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--config",default="configs/assistment2009.yaml")
    parser.add_argument("--split",choices=["train","valid","test"],default="valid")
    parser.add_argument("--max-batches",type=int,default=2)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--output")
    args=parser.parse_args(argv)
    config=load_config(args.config,model_name="dha_kt")
    config["train"]["device"]=args.device
    report=diagnose_checkpoint(config,args.checkpoint,args.split,args.max_batches)
    text=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)
    if args.output:
        path=project_path(args.output)
        if path.resolve()==project_path(args.checkpoint).resolve():
            raise ValueError("Diagnostic output must not overwrite checkpoint.")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="utf-8")
    print(text,flush=True)
    return report


if __name__=="__main__":
    main()
