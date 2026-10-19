"""Stop only residual-condition gradients while preserving normal residual paths."""
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.model import NewQV2
from src.model.newq_v2.diagnostics import diagnose_checkpoint
from src.model.registry import build_model
from src.training.trainer import evaluate_checkpoint, model_kwargs, train
from src.utils.config import load_config
from test_newq import multi_concept_batch


class ResidualConditionDetachTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.kwargs=dict(num_problems=8,num_skills=4,embedding_dim=8,num_heads=2,dropout=0,
                         diffusion_steps=4,attention_embedding_dim=2,residual_embedding_dim=2)
        self.batch=multi_concept_batch()

    def model(self,mode="condition",detach=False):
        torch.manual_seed(173)
        model=NewQV2(**self.kwargs,residual_diffusion_enabled=mode!="off",
                    residual_diffusion_role=mode if mode!="off" else "condition",residual_condition_detach=detach)
        if mode in {"condition","both"}:
            # Zero-initialized condition columns would hide the routing difference.
            with torch.no_grad():
                model.diffusion.denoiser[0].weight[:,-8:].fill_(.1)
        return model

    def test_boolean_validation_default_and_real_config_construction(self):
        self.assertIs(NewQV2(**self.kwargs).residual_condition_detach,False)
        for value in [0,1,"true","false",None,[],np.bool_(True)]:
            with self.assertRaisesRegex(ValueError,"residual_condition_detach must be a bool"):
                NewQV2(**self.kwargs,residual_condition_detach=value)
        config=load_config(model_name="newq_v2")
        self.assertIsInstance(config["model"]["residual_condition_detach"],bool)
        for mode in ["off","condition","target","both"]:
            config["model"].update(residual_diffusion_enabled=mode!="off",
                residual_diffusion_role=mode if mode!="off" else "condition",residual_condition_detach=True)
            model=build_model("newq_v2",model_kwargs(config,{"num_problems":8,"num_skills":4}))
            metadata=model.experiment_metadata()
            self.assertIs(metadata["residual_condition_detach"],True)
            self.assertEqual(metadata["effective_residual_condition_detach"],mode in {"condition","both"})

    def test_no_parameter_or_forward_value_changes_in_any_mode(self):
        for mode in ["off","condition","target","both"]:
            control,detached=self.model(mode,False),self.model(mode,True)
            self.assertEqual(control.state_dict().keys(),detached.state_dict().keys())
            for key,value in control.state_dict().items():
                torch.testing.assert_close(detached.state_dict()[key],value,atol=0,rtol=0)
            self.assertEqual(control.experiment_metadata()["parameter_count"],detached.experiment_metadata()["parameter_count"])
            for training in [True,False]:
                outputs=[]
                for model in [control,detached]:
                    model.train(training)
                    torch.manual_seed(17)
                    outputs.append(model(self.batch,generator=torch.Generator().manual_seed(18)))
                for key,value in outputs[0].items():
                    if value is None:
                        self.assertIsNone(outputs[1][key])
                    else:
                        torch.testing.assert_close(outputs[1][key],value,atol=0,rtol=0,msg=f"{mode}/{key}")

    def test_only_the_condition_view_is_detached_targets_outputs_and_original_stay_live(self):
        for mode in ["condition","both"]:
            model=self.model(mode,True)
            parts=model.question_components(self.batch)
            route=model.diffusion_route(parts)
            torch.testing.assert_close(route["residual_condition"],parts["residual"],atol=0,rtol=0)
            self.assertFalse(route["residual_condition"].requires_grad)
            self.assertTrue(parts["residual"].requires_grad)
            target_name="original" if mode=="both" else "semantic_original"
            self.assertIs(route["diffusion_target"],parts[target_name])
            if mode=="condition":
                with patch.object(model.diffusion,"training_representation",return_value=(torch.zeros_like(parts["original"]),torch.tensor(.2))):
                    enhanced,_=model.enhance_questions(parts,self.batch)
                enhanced.square().sum().backward()
                self.assertGreater(float(model.residual_question_embedding.weight.grad.abs().sum()),0)
            model.zero_grad(set_to_none=True)
            parts=model.question_components(self.batch)
            parts["original"].square().sum().backward()
            self.assertGreater(float(model.residual_question_embedding.weight.grad.abs().sum()),0)

    def test_loss_gradients_block_only_the_condition_route_and_keep_generator_learning(self):
        residual_names={"residual_question_embedding.weight","residual_projection.weight"}
        for mode in ["off","condition","target","both"]:
            control,detached=self.model(mode,False),self.model(mode,True)
            for objective in ["kt_loss","contrastive_loss","diffusion_loss"]:
                gradients=[]
                for model in [control,detached]:
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(17)
                    result=model(self.batch)
                    result[objective].backward()
                    gradients.append({key:parameter.grad.detach().clone() if parameter.grad is not None else None
                                      for key,parameter in model.named_parameters()})
                for name,value in gradients[0].items():
                    actual=gradients[1][name]
                    if name in residual_names and mode in {"condition","both"}:
                        if mode=="condition" and objective=="diffusion_loss":
                            self.assertTrue(actual is None or (actual==0).all())
                            self.assertGreater(float(value.abs().sum()),0)
                        else:
                            self.assertIsNotNone(actual)
                            self.assertGreater(float(actual.abs().sum()),0)
                            self.assertFalse(torch.equal(actual,value))
                    elif value is None:
                        self.assertIsNone(actual)
                    else:
                        torch.testing.assert_close(actual,value,atol=1e-7,rtol=1e-6,msg=f"{mode}/{objective}/{name}")
                if mode in {"condition","both"}:
                    self.assertGreater(float(gradients[1]["diffusion.denoiser.0.weight"][:,-8:].abs().sum()),0)
                if mode=="condition" and objective=="diffusion_loss":
                    saved=[p.detach().clone() for p in [detached.residual_question_embedding.weight,detached.residual_projection.weight]]
                    torch.optim.Adam(detached.parameters(),lr=.002,weight_decay=1e-5).step()
                    for parameter,expected in zip([detached.residual_question_embedding.weight,detached.residual_projection.weight],saved):
                        torch.testing.assert_close(parameter,expected,atol=0,rtol=0)

    def test_detached_condition_still_conveys_residual_values_to_the_generator(self):
        model=self.model("condition",True).eval()
        before=model.question_components(self.batch)
        with torch.no_grad():
            generated_before,_,_=model.generate_questions(before,self.batch,torch.Generator().manual_seed(21))
            model.residual_question_embedding.weight[1:].add_(.5)
        after=model.question_components(self.batch)
        with torch.no_grad():
            generated_after,_,_=model.generate_questions(after,self.batch,torch.Generator().manual_seed(21))
        for key in ["concept","semantic_original","conditioning"]:
            torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
        self.assertFalse(torch.equal(generated_before,generated_after))

    def test_training_restoration_and_diagnostic_persist_flag_without_overriding_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/"metadata.json").write_text('{"num_problems":8,"num_skills":4}',encoding="utf-8")
            for index,split in enumerate(["train","valid","test"]):
                np.savez_compressed(root/f"{split}.npz",problemId=self.batch["problem_ids"].numpy(),skill=self.batch["skill_ids"].numpy(),
                    correct=self.batch["answers"].numpy(),studentId=self.batch["student_ids"].numpy()+index*10,
                    lengths=self.batch["valid_mask"].sum(-1).numpy())
            config=load_config(model_name="newq_v2")
            config["dataset"]["paths"]["processed"]=str(root)
            config["model"].update(embedding_dim=8,dropout=0,diffusion_steps=4,attention_embedding_dim=2,residual_embedding_dim=2,
                residual_diffusion_enabled=True,residual_diffusion_role="condition",residual_condition_detach=True)
            config["train"].update(device="cpu",num_workers=0,num_threads=2,batch_size=2,epochs=1,preprocess_each_run=False,
                                    output_root=str(root/"experiments"))
            with redirect_stdout(io.StringIO()):
                summary=train(config)
            directory=Path(summary["output_dir"])
            self.assertIs(summary["residual_condition_detach"],True)
            self.assertIs(summary["effective_residual_condition_detach"],True)
            saved=yaml.safe_load((directory/"config.yaml").read_text(encoding="utf-8"))
            self.assertIs(saved["model"]["residual_condition_detach"],True)
            checkpoint_path=directory/"best.pt"
            checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
            self.assertIs(checkpoint["model_kwargs"]["residual_condition_detach"],True)
            self.assertIs(checkpoint["residual_condition_detach"],True)
            fingerprint=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            config["model"]["residual_condition_detach"]=False
            built=[]
            def factory(name,kwargs):
                loaded=build_model(name,kwargs)
                built.append(loaded)
                return loaded
            with patch("src.training.trainer.build_model",side_effect=factory):
                evaluate_checkpoint(config,checkpoint_path,verbose=False)
            self.assertIs(built[0].residual_condition_detach,True)
            report=diagnose_checkpoint(config,checkpoint_path,max_batches=1)
            self.assertIs(report["effective_residual_condition_detach"],True)
            self.assertFalse(report["gradient_probe"]["objectives"]["diffusion_loss"]["residual_projection"]["has_gradient"])
            self.assertEqual(hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),fingerprint)
            legacy={**checkpoint,"model_kwargs":checkpoint["model_kwargs"].copy()}
            legacy["model_kwargs"].pop("residual_condition_detach")
            legacy.pop("residual_condition_detach")
            legacy.pop("effective_residual_condition_detach")
            legacy_path=root/"legacy.pt"
            torch.save(legacy,legacy_path)
            old=build_model("newq_v2",legacy["model_kwargs"])
            old.load_state_dict(legacy["model"],strict=True)
            self.assertIs(old.residual_condition_detach,False)
            config["model"]["residual_condition_detach"]=True
            old_report=diagnose_checkpoint(config,legacy_path,max_batches=1)
            self.assertIs(old_report["residual_condition_detach"],False)


if __name__=="__main__":
    unittest.main()
