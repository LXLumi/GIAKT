"""Mixture arithmetic, query/value separation, candidate stability and task gradients."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch

from src.model import DHAKT
from src.model.dha_kt.evidence import CandidatePredictor, candidate_moments, mixture_logit, responsibilities
from src.model.dha_kt.candidates import build_candidates
from dha_kt_helpers import prepared_model
from test_newq import multi_concept_batch


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)

    def test_hand_computed_posterior_and_probability_mixture(self):
        logits = torch.logit(torch.tensor([[[.8, .4]]]))
        prior = torch.tensor([.5,.5]).log()
        valid = torch.tensor([[True]])
        wrong = responsibilities(logits, torch.tensor([[0]]), prior, valid)
        correct = responsibilities(logits, torch.tensor([[1]]), prior, valid)
        torch.testing.assert_close(wrong, torch.tensor([[[.25,.75]]]))
        torch.testing.assert_close(correct, torch.tensor([[[2/3,1/3]]]))
        torch.testing.assert_close(mixture_logit(logits, prior).sigmoid(), torch.tensor([[.6]]))
        self.assertFalse(wrong.requires_grad)
        extreme = mixture_logit(torch.tensor([[-10000.,10000.]]), prior)
        self.assertTrue(torch.isfinite(extreme).all())
        self.assertEqual(float(extreme), 0.)

    def test_shared_dropout_identical_candidates_and_payload_gradients(self):
        head = CandidatePredictor(8, .5).train()
        candidate = torch.randn(2,4,1,8).expand(2,4,4,8)
        scores = head(torch.randn(2,4,8), candidate)
        torch.testing.assert_close(scores, scores[..., :1].expand_as(scores), atol=0, rtol=0)
        payload = torch.randn(2,4,4,8, requires_grad=True)
        logits = torch.randn(2,4,4, requires_grad=True)
        weights = responsibilities(logits, torch.ones(2,4,dtype=torch.long), torch.full((4,),.25).log(), torch.ones(2,4,dtype=torch.bool))
        mean, variance = candidate_moments(payload, weights)
        (mean.square().sum()+variance.sum()).backward()
        self.assertIsNone(logits.grad)
        self.assertGreater(float(payload.grad.abs().sum()),0)
        mean, variance = candidate_moments(candidate, weights)
        torch.testing.assert_close(mean, candidate[...,0,:], atol=0, rtol=0)
        self.assertTrue((variance==0).all())

    def test_current_future_labels_and_first_history_are_causal(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_ = prepared_model(temporary)
            model.eval()
            with torch.no_grad():
                model.adapter.weight.normal_(std=.1)
                model.profile_to_residual.weight.mul_(20)
            seed = lambda: torch.Generator().manual_seed(71)
            expected = model(batch,generator=seed(),return_details=True)
            for position in [0,1,2]:
                changed = {k:v.clone() for k,v in batch.items()}
                changed["answers"][:,position] = 1-changed["answers"][:,position]
                actual = model(changed,generator=seed(),return_details=True)
                torch.testing.assert_close(actual["logits"][:,:position+1],expected["logits"][:,:position+1],atol=1e-6,rtol=1e-6)
                if position:
                    torch.testing.assert_close(actual["details"]["responsibilities"][:,:position],expected["details"]["responsibilities"][:,:position],atol=0,rtol=0)
                self.assertFalse(torch.equal(actual["logits"][:,position+1:],expected["logits"][:,position+1:]))
            changed = {k:v.clone() for k,v in batch.items()}
            changed["problem_ids"][:,3:] = 8
            changed["skill_ids"][:,3:] = 4
            actual = model(changed,generator=seed())
            torch.testing.assert_close(actual["logits"][:,:3], expected["logits"][:,:3],atol=1e-6,rtol=1e-6)

    def test_second_pass_queries_are_original_and_interactions_carry_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_ = prepared_model(temporary)
            model.eval()
            with torch.no_grad():
                model.adapter.weight.fill_(.1)
            captured=[]
            original = model.retriever.forward_with_interactions
            def spy(q, x, valid, context=None):
                captured.append((q.detach().clone(),x.detach().clone()))
                return original(q,x,valid,context)
            with patch.object(model.retriever,"forward_with_interactions",side_effect=spy):
                output = model(batch,return_details=True)
            self.assertEqual(len(captured),2)
            torch.testing.assert_close(captured[0][0], output["details"]["q0"])
            torch.testing.assert_close(captured[1][0], output["details"]["q0"])
            self.assertFalse(torch.equal(captured[0][1],captured[1][1]))

    def test_identical_candidates_degenerate_and_disabled_R_has_no_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_ = prepared_model(temporary,generated_correction_enabled=False,learned_residual_enabled=False)
            model.eval()
            with torch.no_grad():
                model.adapter.weight.normal_()
            output=model(batch,return_details=True)
            details=output["details"]
            torch.testing.assert_close(details["x1"],details["x0"],atol=0,rtol=0)
            torch.testing.assert_close(output["states"],details["h0"],atol=0,rtol=0)
            self.assertFalse(any("residual_code" in n or "residual_projection" in n for n,_ in model.named_parameters()))
            self.assertTrue((model.item_encoder(batch)["residual"]==0).all())

    def test_prior_ablation_changes_history_weights_without_reading_current_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_=prepared_model(temporary,responsibility_mode="prior")
            model.eval()
            result=model(batch,return_details=True)
            expected=model.log_prior.exp().expand_as(result["details"]["responsibilities"])*batch["valid_mask"].unsqueeze(-1)
            torch.testing.assert_close(result["details"]["responsibilities"],expected,atol=0,rtol=0)

    def test_all_fusions_injection_positions_masks_and_no_evidence_fallback(self):
        for fusion,keep in [("add",False),("concat",False),("concat",True)]:
            for injection in ["residual_update","semantic_update"]:
                with tempfile.TemporaryDirectory() as temporary:
                    model,batch,_=prepared_model(temporary,residual_fusion=fusion,residual_concat_keep_2d=keep,profile_injection=injection)
                    batch={k:v.clone() for k,v in batch.items()}
                    batch["valid_mask"][1,2:]=False
                    batch["prediction_mask"][1,2:]=False
                    batch["problem_ids"][1,2:]=8
                    batch["skill_ids"][1,2:]=4
                    output=model(batch,return_details=True)
                    self.assertEqual(output["states"].shape[-1],16 if keep else 8)
                    self.assertTrue((output["states"][~batch["valid_mask"]]==0).all())
                    self.assertTrue((output["details"]["candidates"][~batch["valid_mask"]]==0).all())
                    parts=model.item_encoder(batch)
                    delta=torch.full_like(parts["residual"],.2)
                    expected=model.item_encoder.fuse(parts["semantic"]+(delta if injection=="semantic_update" else 0),
                                                     parts["residual"]+(delta if injection=="residual_update" else 0))
                    self.assertEqual(expected.shape[-1],model.question_dim)
                    latent=torch.full(parts["semantic"].shape[:-1]+(4,),.1)
                    delta=model.profile_to_residual(latent)
                    expected=model.item_encoder.fuse(parts["semantic"]+(delta if injection=="semantic_update" else 0),
                                                     parts["residual"]+(delta if injection=="residual_update" else 0))
                    expected=expected*batch["valid_mask"].unsqueeze(-1)
                    with patch.object(model.diffusion,"sample",side_effect=lambda structure,partial,*args,**kwargs:torch.full_like(partial,.4)):
                        actual=build_candidates(model,parts,batch)
                    torch.testing.assert_close(actual[...,1:,:],expected.unsqueeze(-2).expand_as(actual[...,1:,:]))
                    with torch.no_grad():
                        model.reference_bank.partial_observed.zero_()
                        model.reference_bank.partial_counts.zero_()
                        model.reference_bank.partial_profile.zero_()
                    candidates=build_candidates(model,model.item_encoder(batch),batch)
                    torch.testing.assert_close(candidates,candidates[...,:1,:].expand_as(candidates),atol=0,rtol=0)

    def test_actual_sampler_and_pilot_train_generator_without_detaching_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_=prepared_model(temporary)
            for objective in ["kt_loss","pilot_loss","diffusion_loss"]:
                model.zero_grad(set_to_none=True)
                output=model(batch,return_details=True)
                output[objective].backward()
                norm=sum(float(p.grad.abs().sum()) for p in model.diffusion.parameters() if p.grad is not None)
                self.assertGreater(norm,0)
                if objective!="diffusion_loss":
                    self.assertGreater(float(model.profile_to_residual.weight.grad.abs().sum()),0)
                self.assertFalse(model.reference_bank.dense_profile.requires_grad)
                self.assertFalse(output["details"]["responsibilities"].requires_grad)

    def test_stable_candidates_across_batch_order_generator_and_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_=prepared_model(temporary)
            model.eval()
            with torch.no_grad():
                first=model(batch,generator=torch.Generator().manual_seed(101),return_details=True)
                reordered={k:v.flip(0) for k,v in batch.items()}
                second=model(reordered,generator=torch.Generator().manual_seed(101),return_details=True)
                torch.testing.assert_close(first["details"]["candidates"],second["details"]["candidates"].flip(0),atol=1e-7,rtol=1e-6)
                single=model({k:v[:1] for k,v in batch.items()},generator=torch.Generator().manual_seed(101),return_details=True)
                torch.testing.assert_close(first["logits"][:1],single["logits"],atol=1e-6,rtol=1e-6)
                alternative=model(batch,generator=torch.Generator().manual_seed(102),return_details=True)
                self.assertFalse(torch.equal(first["details"]["candidates"],alternative["details"]["candidates"]))
            model.train()
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            model.zero_grad(set_to_none=True)
            before=model(batch,return_details=True)
            before["loss"].backward()
            optimizer.step()
            after=model(batch,return_details=True)
            self.assertFalse(torch.equal(before["details"]["candidates"],after["details"]["candidates"]))

    def test_legacy_routes_fusion_once_and_target_candidates_ignore_current_R(self):
        batch=multi_concept_batch()
        for fusion,keep in [("add",False),("concat",False),("concat",True)]:
            for mode in ["off","condition","target","both"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model=DHAKT(8,4,embedding_dim=8,dropout=0,attention_embedding_dim=2,residual_embedding_dim=2,
                        candidate_backend="legacy_question",profile_injection=None,legacy_residual_mode=mode,
                        residual_fusion=fusion,residual_concat_keep_2d=keep,diffusion={"steps":4})
                model.eval()
                before=model(batch,generator=torch.Generator().manual_seed(67),return_details=True)
                self.assertEqual(before["details"]["candidates"].shape[-1],16 if keep else 8)
                with torch.no_grad():
                    model.item_encoder.residual_code.weight[1:].add_(.4)
                after=model(batch,generator=torch.Generator().manual_seed(67),return_details=True)
                if mode=="target":
                    torch.testing.assert_close(before["details"]["candidates"][...,1:,:],after["details"]["candidates"][...,1:,:],atol=0,rtol=0)
                self.assertFalse(torch.equal(before["details"]["q0"],after["details"]["q0"]))
                model.train()
                model(batch)["loss"].backward()


if __name__=="__main__":
    unittest.main()
