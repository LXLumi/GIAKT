"""Section 15: residual condition/target routing, initialization and saved-mode recovery."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
import yaml

from src.model import NewQV2
from src.model.diffuqkt.diffusion import ConditionalQuestionDiffusion
from src.model.newq_v2.diffusion import ResidualConditionDiffusion
from src.model.newq_v2.diagnostics import diagnose_checkpoint
from src.model.registry import build_model
from src.training.trainer import evaluate_checkpoint, model_kwargs, train
from src.utils.config import load_config
from src.utils.files import sha256
from test_newq import multi_concept_batch


MODES = ["off", "condition", "target", "both"]


class ResidualPluginTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.kwargs = dict(num_problems=8,num_skills=4,embedding_dim=8,num_heads=2,dropout=0,
                           diffusion_steps=4,attention_embedding_dim=2,residual_embedding_dim=2)
        self.batch = multi_concept_batch()

    def model(self, mode, **extra):
        torch.manual_seed(173)
        return NewQV2(**{**self.kwargs,"residual_diffusion_enabled":mode!="off",
                         "residual_diffusion_role":mode if mode!="off" else "condition",**extra})

    @staticmethod
    def activate_condition(model):
        if isinstance(model.diffusion,ResidualConditionDiffusion):
            with torch.no_grad():
                model.diffusion.denoiser[0].weight[:, -8:].fill_(.1)

    def test_switch_validation_yaml_defaults_and_complete_construction_chain(self):
        for enabled in [0,1,"true","false",None,[],np.bool_(True)]:
            with self.assertRaisesRegex(ValueError,"must be a bool"):
                NewQV2(**self.kwargs,residual_diffusion_enabled=enabled)
        for enabled in [False,True]:
            for role in [None,"Condition","off","unknown",True,[],1]:
                with self.assertRaisesRegex(ValueError,"residual_diffusion_role"):
                    NewQV2(**self.kwargs,residual_diffusion_enabled=enabled,residual_diffusion_role=role)
        config=load_config(model_name="newq_v2")
        self.assertIsInstance(config["model"]["residual_diffusion_enabled"],bool)
        self.assertIn(config["model"]["residual_diffusion_role"],{"condition","target","both"})
        model=build_model("newq_v2",model_kwargs(config,{"num_problems":8,"num_skills":4}))
        expected_mode=config["model"]["residual_diffusion_role"] if config["model"]["residual_diffusion_enabled"] else "off"
        self.assertEqual(model.effective_residual_diffusion_mode,expected_mode)
        self.assertEqual(NewQV2(**self.kwargs).effective_residual_diffusion_mode,"off")

    def test_disabled_roles_are_identical_and_condition_expansion_preserves_common_initialization(self):
        base=self.model("off")
        state=base.state_dict()
        expected_rng=torch.random.get_rng_state().clone()
        for mode in MODES:
            model=self.model(mode)
            torch.testing.assert_close(torch.random.get_rng_state(),expected_rng,atol=0,rtol=0)
            for key,value in state.items():
                actual=model.state_dict()[key]
                if key=="diffusion.denoiser.0.weight" and mode in {"condition","both"}:
                    torch.testing.assert_close(actual[:,:25],value,atol=0,rtol=0)
                    self.assertTrue((actual[:,25:]==0).all())
                else:
                    torch.testing.assert_close(actual,value,atol=0,rtol=0,msg=key)
            self.assertEqual(len([name for name,_ in model.named_children() if name=="diffusion"]),1)
        base.eval()
        expected=base(self.batch,generator=torch.Generator().manual_seed(11))
        for role in ["condition","target","both"]:
            model=self.model("off",residual_diffusion_role=role).eval()
            self.assertIs(type(model.diffusion),ConditionalQuestionDiffusion)
            actual=model(self.batch,generator=torch.Generator().manual_seed(11))
            torch.testing.assert_close(actual["logits"],expected["logits"],atol=0,rtol=0)

    def test_fixed_generation_spies_capture_targets_conditions_and_single_output_addition(self):
        for mode in MODES:
            model=self.model(mode)
            parts=model.question_components(self.batch)
            generated=torch.full_like(parts["original"],2)
            expected_target=parts["original"] if mode in {"target","both"} else parts["semantic_original"]
            expected=(generated if mode in {"target","both"} else generated+parts["residual"])*self.batch["valid_mask"].unsqueeze(-1)
            for training in [True,False]:
                model.train(training)
                method="training_representation" if training else "sample"
                returned=(generated,torch.tensor(.7)) if training else generated
                generator=torch.Generator().manual_seed(12)
                with patch.object(model.diffusion,method,return_value=returned) as spy:
                    actual,losses=model.enhance_questions(parts,self.batch,generator)
                torch.testing.assert_close(actual,expected)
                args=spy.call_args.args
                offset=1 if training else 0
                if training:
                    self.assertIs(args[0],expected_target)
                self.assertIs(args[offset],parts["concept"])
                self.assertIs(args[offset+1],parts["conditioning"])
                if mode in {"condition","both"}:
                    self.assertIs(spy.call_args.kwargs["residual_condition"],parts["residual"])
                else:
                    self.assertEqual(spy.call_args.kwargs,{})
                if not training:
                    self.assertIs(args[3],generator)
                    self.assertIsNone(losses["diffusion_loss"])
                torch.testing.assert_close(parts["original"],parts["semantic_original"]+parts["residual"])
                torch.testing.assert_close(parts["concept"],parts["fused_concept"])

    def test_condition_affects_denoising_and_is_passed_to_every_step_with_standard_noise(self):
        base=self.model("off")
        parts=base.question_components(self.batch)
        noisy=torch.randn_like(parts["concept"])
        time=torch.zeros_like(self.batch["problem_ids"])
        for mode in ["condition","both"]:
            model=self.model(mode)
            diffusion=model.diffusion
            zeros=torch.zeros_like(parts["residual"])
            torch.testing.assert_close(diffusion.predict_x0(noisy,time,parts["concept"],parts["conditioning"],zeros),
                base.diffusion.predict_x0(noisy,time,parts["concept"],parts["conditioning"]),atol=5e-7,rtol=1e-6)
            self.activate_condition(model)
            a=diffusion.predict_x0(noisy,time,parts["concept"],parts["conditioning"],zeros)
            b=diffusion.predict_x0(noisy,time,parts["concept"],parts["conditioning"],zeros+1)
            self.assertFalse(torch.equal(a,b))
            condition=parts["residual"].detach().clone()
            condition[~self.batch["valid_mask"]]=999
            with patch.object(diffusion,"predict_x0",wraps=diffusion.predict_x0) as spy:
                diffusion.sample(parts["concept"],parts["conditioning"],self.batch["valid_mask"],
                                 torch.Generator().manual_seed(13),residual_condition=condition)
            self.assertEqual(spy.call_count,4)
            calls=spy.call_args_list
            received=calls[0].kwargs["residual_condition"]
            self.assertTrue((received[~self.batch["valid_mask"]]==0).all())
            for call in calls:
                self.assertIs(call.kwargs["residual_condition"],received)
            expected_noise=torch.randn(parts["concept"].shape,generator=torch.Generator().manual_seed(13))*self.batch["valid_mask"].unsqueeze(-1)
            torch.testing.assert_close(calls[0].args[0],expected_noise,atol=0,rtol=0)
        for mode in ["off","target"]:
            model=self.model(mode)
            with patch.object(model.diffusion,"predict_x0",wraps=model.diffusion.predict_x0) as spy:
                model.eval().enhance_questions(parts,self.batch,torch.Generator().manual_seed(13))
            self.assertEqual(spy.call_count,4)
            self.assertTrue(all(not call.kwargs for call in spy.call_args_list))
            with self.assertRaises(TypeError):
                model.diffusion.predict_x0(noisy,time,parts["concept"],parts["conditioning"],residual_condition=zeros)

    def test_original_sampling_equations_and_variance_modes_remain_equivalent_for_zero_condition(self):
        for variance in ["posterior","fixed_large"]:
            base=self.model("off",reverse_variance=variance)
            expanded=self.model("condition",reverse_variance=variance)
            parts=base.question_components(self.batch)
            valid=self.batch["valid_mask"]
            expected=base.diffusion.sample(parts["concept"],parts["conditioning"],valid,torch.Generator().manual_seed(14))
            actual=expanded.diffusion.sample(parts["concept"],parts["conditioning"],valid,torch.Generator().manual_seed(14),
                                            residual_condition=torch.zeros_like(parts["residual"]))
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

    def test_target_evaluation_ignores_current_residual_but_its_training_noise_changes(self):
        for mode in MODES:
            model=self.model(mode)
            self.activate_condition(model)
            model.eval()
            before=model.question_components(self.batch)
            with torch.no_grad():
                predicted_before=model(self.batch,generator=torch.Generator().manual_seed(15))
                model.residual_question_embedding.weight[1:].add_(.5)
            after=model.question_components(self.batch)
            with torch.no_grad():
                predicted_after=model(self.batch,generator=torch.Generator().manual_seed(15))
            if mode=="target":
                torch.testing.assert_close(predicted_after["logits"],predicted_before["logits"],atol=0,rtol=0)
            else:
                self.assertFalse(torch.equal(predicted_after["logits"],predicted_before["logits"]))
            model.train()
            noisy=[]
            for parts in [before,after]:
                torch.manual_seed(16)
                with patch.object(model.diffusion,"q_sample",wraps=model.diffusion.q_sample) as spy:
                    model.enhance_questions(parts,self.batch)
                noisy.append(spy.call_args.args[0])
            if mode in {"target","both"}:
                self.assertFalse(torch.equal(noisy[0],noisy[1]))
            else:
                torch.testing.assert_close(noisy[0],noisy[1],atol=0,rtol=0)

    def test_gradient_routes_and_zero_condition_initialization_are_distinct(self):
        for mode in MODES:
            model=self.model(mode)
            if mode=="condition":
                torch.manual_seed(17)
                model(self.batch)["diffusion_loss"].backward()
                gradient=model.residual_question_embedding.weight.grad
                self.assertTrue(gradient is None or (gradient==0).all())
                self.assertGreater(float(model.diffusion.denoiser[0].weight.grad[:,25:].abs().sum()),0)
                model.zero_grad(set_to_none=True)
            self.activate_condition(model)
            for objective in ["kt_loss","contrastive_loss","diffusion_loss"]:
                model.zero_grad(set_to_none=True)
                torch.manual_seed(17)
                result=model(self.batch)
                result[objective].backward()
                for module in [model.residual_question_embedding,model.residual_projection]:
                    gradient=module.weight.grad
                    if mode=="off" and objective=="diffusion_loss":
                        self.assertTrue(gradient is None or (gradient==0).all())
                    else:
                        self.assertIsNotNone(gradient)
                        self.assertTrue(torch.isfinite(gradient).all())
                        self.assertGreater(float(gradient.abs().sum()),0)
                for embedding in [model.attention_question_embedding,model.residual_question_embedding,model.concept,model.variation,model.difficulty]:
                    if embedding.weight.grad is not None:
                        self.assertTrue((embedding.weight.grad[0]==0).all())
            torch.testing.assert_close(result["loss"],result["kt_loss"]+.001*result["contrastive_loss"]+.01*result["diffusion_loss"])

    def test_contrastive_objects_and_causality_in_all_modes(self):
        for mode in MODES:
            model=self.model(mode)
            self.activate_condition(model)
            parts=model.question_components(self.batch)
            generated=parts["semantic_original"]*.3
            captured=[]
            handle=model.retriever.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
            try:
                with patch.object(model.diffusion,"training_representation",return_value=(generated,torch.tensor(.4))):
                    model(self.batch)
            finally:
                handle.remove()
            torch.testing.assert_close(captured[1],parts["original"])
            expected=generated if mode in {"target","both"} else generated+parts["residual"]
            torch.testing.assert_close(captured[0],expected)
            model.eval()
            expected=model(self.batch,generator=torch.Generator().manual_seed(18))
            changed={key:value.clone() for key,value in self.batch.items()}
            changed["answers"][:,2:]=1-changed["answers"][:,2:]
            actual=model(changed,generator=torch.Generator().manual_seed(18))
            torch.testing.assert_close(actual["logits"][:,:3],expected["logits"][:,:3])
            changed={key:value.clone() for key,value in self.batch.items()}
            changed["problem_ids"][:,3:]=8
            changed["skill_ids"][:,3:]=4
            actual=model(changed,generator=torch.Generator().manual_seed(18))
            torch.testing.assert_close(actual["logits"][:,:3],expected["logits"][:,:3])
            changed={key:value.clone() for key,value in self.batch.items()}
            changed["problem_ids"][~self.batch["valid_mask"]]=8
            changed["skill_ids"][~self.batch["valid_mask"]]=4
            actual=model(changed,generator=torch.Generator().manual_seed(18))
            torch.testing.assert_close(actual["logits"][self.batch["valid_mask"]],expected["logits"][self.batch["valid_mask"]])

    def test_condition_validation_and_actual_full_parameter_budgets(self):
        model=self.model("condition")
        parts=model.question_components(self.batch)
        for residual in [None,torch.zeros(2,5,7),torch.zeros(2,4,8),parts["residual"].double(),torch.zeros(2,5,8,device="meta")]:
            with self.assertRaisesRegex(ValueError,"Residual condition"):
                model.diffusion.sample(parts["concept"],parts["conditioning"],self.batch["valid_mask"],residual_condition=residual)
        for problems,skills,off_count,on_count in [(17645,123,830549,863317),(2210,101,562522,595290)]:
            for mode in MODES:
                model=NewQV2(problems,skills,residual_diffusion_enabled=mode!="off",residual_diffusion_role=mode if mode!="off" else "condition")
                self.assertEqual(sum(p.numel() for p in model.parameters()),on_count if mode in {"condition","both"} else off_count)
                self.assertLess(model.parameter_ratio,3)
        base=NewQV2(10000,4)
        total_dim=int(3*base.baseline_parameter_count/(10000+129))
        options=dict(attention_embedding_dim=total_dim//2,residual_embedding_dim=total_dim-total_dim//2)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            off=NewQV2(10000,4,**options)
        self.assertTrue(3<off.parameter_ratio<=4)
        self.assertTrue(any("3x target" in str(item.message) for item in captured))
        with self.assertRaisesRegex(ValueError,"4x baseline"):
            NewQV2(10000,4,**options,residual_diffusion_enabled=True)

    def create_data(self,root):
        (root/"metadata.json").write_text(json.dumps({"num_problems":8,"num_skills":4}),encoding="utf-8")
        for index,split in enumerate(["train","valid","test"]):
            np.savez_compressed(root/f"{split}.npz",problemId=self.batch["problem_ids"].numpy(),skill=self.batch["skill_ids"].numpy(),
                correct=self.batch["answers"].numpy(),studentId=self.batch["student_ids"].numpy()+index*10,
                lengths=self.batch["valid_mask"].sum(-1).numpy())
        config=load_config(model_name="newq_v2")
        config["dataset"]["paths"]["processed"]=str(root)
        config["model"].update(embedding_dim=8,dropout=0,diffusion_steps=4,attention_embedding_dim=2,residual_embedding_dim=2)
        config["train"].update(device="cpu",batch_size=2,num_workers=0,num_threads=2,epochs=1,preprocess_each_run=False)
        return config

    def test_four_mode_checkpoint_evaluation_and_readonly_diagnostics_use_saved_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=self.create_data(root)
            for mode in MODES+["legacy"]:
                actual_mode="off" if mode=="legacy" else mode
                model=self.model(actual_mode)
                kwargs={**self.kwargs,"residual_diffusion_enabled":actual_mode!="off", "residual_diffusion_role":actual_mode if actual_mode!="off" else "condition"}
                if mode=="legacy":
                    kwargs=self.kwargs.copy()
                path=root/f"{mode}.pt"
                torch.save({"model_name":"newq_v2","model":model.state_dict(),"model_kwargs":kwargs,"config":config,
                            "epoch":1,"metadata_sha256":sha256(root/"metadata.json")},path)
                before=hashlib.sha256(path.read_bytes()).hexdigest()
                built=[]
                def factory(name,arguments):
                    loaded=build_model(name,arguments)
                    built.append(loaded)
                    return loaded
                config["model"].update(residual_diffusion_enabled=actual_mode in {"off","condition","both"},
                                        residual_diffusion_role="both" if actual_mode=="condition" else "condition")
                with patch("src.training.trainer.build_model",side_effect=factory):
                    evaluate_checkpoint(config,path,verbose=False)
                self.assertEqual(built[0].effective_residual_diffusion_mode,actual_mode)
                report=diagnose_checkpoint(config,path,max_batches=1)
                self.assertEqual(report["effective_residual_diffusion_mode"],actual_mode)
                self.assertEqual(report["has_residual_condition"],actual_mode in {"condition","both"})
                self.assertEqual(report["add_residual_after_generation"],actual_mode not in {"target","both"})
                expected_name="original_generated_norm" if actual_mode in {"target","both"} else "semantic_generated_norm"
                self.assertIn(expected_name,report["statistics"])
                self.assertEqual(report["diffusion_target_name"],"original" if actual_mode in {"target","both"} else "semantic_original")
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_training_persists_requested_flags_effective_mode_and_automatic_run_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=self.create_data(root)
            config["train"]["output_root"]=str(root/"experiments")
            for mode in MODES:
                config["model"].update(residual_diffusion_enabled=mode!="off",residual_diffusion_role=mode if mode!="off" else "condition")
                if mode=="off":
                    # A direct legacy config must also save its normalized default switches.
                    config["model"].pop("residual_diffusion_enabled")
                    config["model"].pop("residual_diffusion_role")
                output=io.StringIO()
                with redirect_stdout(output):
                    summary=train(config)
                self.assertEqual(summary["effective_residual_diffusion_mode"],mode)
                self.assertIn('"effective_residual_diffusion_mode"',output.getvalue())
                directory=Path(summary["output_dir"])
                saved=yaml.safe_load((directory/"config.yaml").read_text(encoding="utf-8"))
                self.assertEqual(saved["model"]["residual_diffusion_enabled"],mode!="off")
                for filename in ["best.pt","last.pt"]:
                    checkpoint=torch.load(directory/filename,map_location="cpu",weights_only=False)
                    self.assertEqual(checkpoint["effective_residual_diffusion_mode"],mode)
                    self.assertEqual(checkpoint["model_kwargs"]["residual_diffusion_role"],mode if mode!="off" else "condition")
                    self.assertEqual(checkpoint["residual_diffusion_enabled"],mode!="off")


if __name__=="__main__":
    unittest.main()
