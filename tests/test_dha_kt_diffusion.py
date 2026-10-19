"""DDIM jump arithmetic, fixed targets, missing supervision and residual-condition gradients."""
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.model.dha_kt.diffusion import ProfileDiffusion
from dha_kt_helpers import prepared_model


class ProfileDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_ddim_jump_formula_eta_and_final_x0(self):
        for eta in [0., .5]:
            model=ProfileDiffusion(4,5,0,steps=11,sampling_steps=4,ddim_eta=eta,time_embedding_dim=8,hidden_dim=16)
            structure,partial,counts,observed=torch.randn(2,5),torch.randn(2,4),torch.ones(2,4),torch.ones(2,4,dtype=torch.bool)
            initial=torch.randn(2,4)
            noises=[torch.randn(2,4) for _ in range(3)]
            calls=[]
            def predictor(noisy,time,*args):
                calls.append(time[0].item())
                return noisy*.1+partial*.2
            state=initial
            for index,s in enumerate(model.sampling_grid.tolist()):
                x0=state*.1+partial*.2
                if index==3:
                    expected=x0
                    break
                r=int(model.sampling_grid[index+1])
                a_s,a_r=model.alpha_bar[s],model.alpha_bar[r]
                eps=(state-a_s.sqrt()*x0)/(1-a_s).sqrt()
                sigma=eta*(((1-a_r)/(1-a_s))*(1-a_s/a_r)).sqrt()
                state=a_r.sqrt()*x0+(1-a_r-sigma.square()).sqrt()*eps
                if eta:
                    state=state+sigma*noises[index]
            with patch.object(model,"predict_x0",side_effect=predictor):
                actual=model.sample(structure,partial,counts,observed,initial,step_noise=lambda i:noises[i])
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            self.assertEqual(calls,model.sampling_grid.tolist())
            self.assertEqual(calls[0],10)
            self.assertEqual(len(calls),len(set(calls)))

    def test_cosine_terminal_snr_and_sampler_is_differentiable(self):
        model=ProfileDiffusion(4,5,0,steps=100,sampling_steps=8,time_embedding_dim=8,hidden_dim=16)
        self.assertGreater(float(model.alpha_bar[-1]),0)
        self.assertLess(float(model.alpha_bar[-1]),1e-5)
        values=model.sample(torch.randn(2,5),torch.zeros(2,4),torch.ones(2,4),torch.ones(2,4,dtype=torch.bool),torch.randn(2,4))
        self.assertTrue(values.requires_grad)
        values.square().sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)

    def test_fixed_noisy_source_masked_loss_and_all_missing_zero(self):
        model=ProfileDiffusion(4,5,0,steps=4,sampling_steps=2,time_embedding_dim=8,hidden_dim=16)
        dense=torch.ones(2,4,requires_grad=True)
        observed=torch.tensor([[True,False,False,False],[False,False,False,False]])
        counts=observed.float()*3
        structure=torch.randn(2,5)
        loss=model.denoising_loss(dense,counts,observed,structure,torch.zeros(2,4),counts,observed)
        loss.backward()
        self.assertIsNone(dense.grad)
        model.zero_grad(set_to_none=True)
        empty=model.denoising_loss(dense,counts*0,observed&False,structure,torch.zeros(2,4),counts*0,observed&False)
        self.assertEqual(float(empty),0.)
        empty.backward()
        self.assertTrue(all(p.grad is None or (p.grad==0).all() for p in model.parameters()))

    def test_condition_detach_only_removes_profile_condition_gradient(self):
        for condition in ["none","detach","trainable"]:
            with tempfile.TemporaryDirectory() as temporary:
                model,batch,_=prepared_model(temporary,profile_residual_condition=condition)
                result=model(batch)
                result["diffusion_loss"].backward()
                gradient=model.item_encoder.residual_code.weight.grad
                if condition=="trainable":
                    self.assertIsNotNone(gradient)
                    self.assertGreater(float(gradient.abs().sum()),0)
                else:
                    self.assertIsNone(gradient)
                model.zero_grad(set_to_none=True)
                model(batch)["kt_loss"].backward()
                self.assertGreater(float(model.item_encoder.residual_code.weight.grad.abs().sum()),0)

    def test_clean_dense_evidence_is_not_used_for_KT_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_=prepared_model(temporary)
            model.eval()
            first=model(batch,generator=torch.Generator().manual_seed(43),return_details=True)
            with torch.no_grad():
                model.reference_bank.dense_profile.add_(100)
                model.reference_bank.dense_counts.mul_(10)
            second=model(batch,generator=torch.Generator().manual_seed(43),return_details=True)
            torch.testing.assert_close(first["details"]["candidates"],second["details"]["candidates"],atol=0,rtol=0)


if __name__=="__main__":
    unittest.main()
