"""Stages, optimizer membership, self-contained evaluation and exact CPU resume."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import torch

from src.model import DHAKT
from src.model.registry import build_model
from src.training.trainer import train,evaluate_checkpoint
from dha_kt_helpers import make_data,prepared_model


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_stage_freezing_and_future_unfreezing_keep_optimizer_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_=prepared_model(temporary,stages=dict(base_warmup_epochs=1,diffusion_pretrain_epochs=1))
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            parameter_ids={id(p) for group in optimizer.param_groups for p in group['params']}
            initial_item=model.item_encoder.concept.weight.detach().clone()
            initial_diff=model.diffusion.denoiser[0].weight.detach().clone()
            for epoch,expected in [(1,'base_warmup'),(2,'diffusion_pretrain'),(3,'joint')]:
                model.on_train_epoch_start(epoch)
                self.assertEqual(model.training_stage,expected)
                self.assertTrue(all(id(p) in parameter_ids for p in model.parameters()))
                before_item=model.item_encoder.concept.weight.detach().clone()
                before_diff=model.diffusion.denoiser[0].weight.detach().clone()
                optimizer.zero_grad(set_to_none=True)
                model(batch)['loss'].backward()
                optimizer.step()
                model.on_optimizer_step()
                if epoch==1:
                    torch.testing.assert_close(model.diffusion.denoiser[0].weight,before_diff,atol=0,rtol=0)
                    self.assertFalse(torch.equal(model.item_encoder.concept.weight,before_item))
                elif epoch==2:
                    torch.testing.assert_close(model.item_encoder.concept.weight,before_item,atol=0,rtol=0)
                    self.assertFalse(torch.equal(model.diffusion.denoiser[0].weight,before_diff))
                else:
                    self.assertFalse(torch.equal(model.item_encoder.concept.weight,before_item))
                    self.assertFalse(torch.equal(model.diffusion.denoiser[0].weight,before_diff))
            self.assertEqual(int(model.global_step),3)

    def test_checkpoint_selfcontained_and_current_config_cannot_change_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=make_data(root)
            config['model'].update(residual_fusion='concat',residual_concat_keep_2d=True)
            with redirect_stdout(io.StringIO()):
                summary=train(config)
            path=Path(summary['output_dir'])/'best.pt'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            for file in (root/'reference').iterdir():
                file.unlink()
            config['model'].update(residual_fusion='add',residual_concat_keep_2d=False)
            config['model']['profile']['artifact_path']='missing'
            config['model']['diffusion']['steps']=9
            metrics=evaluate_checkpoint(config,path,verbose=False)
            for key in ['auc','accuracy','bce','loss']:
                self.assertEqual(metrics[key],summary['test'][key])
            restored=build_model('dha_kt',checkpoint['model_kwargs'])
            restored.load_state_dict(checkpoint['model'],strict=True)
            self.assertEqual(restored.question_dim,16)
            self.assertEqual(restored.training_stage,'joint')
            self.assertTrue(bool(restored.reference_bank.ready))
            self.assertIsNone(summary['train']['contrastive_loss'])
            self.assertIsNotNone(summary['train']['pilot_loss'])
            self.assertIsNone(summary['test']['pilot_loss'])

    def test_cpu_resume_matches_uninterrupted_stages_parameters_and_adam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=make_data(root)
            config['model']['dropout']=.2
            config['model']['stages']=dict(base_warmup_epochs=1,diffusion_pretrain_epochs=1)
            config['train']['epochs']=4
            with redirect_stdout(io.StringIO()):
                full=train(config)
            partial=copy.deepcopy(config)
            partial['train']['epochs']=2
            with redirect_stdout(io.StringIO()):
                part=train(partial)
            continuation=copy.deepcopy(config)
            continuation['train']['resume_checkpoint']=str(Path(part['output_dir'])/'last.pt')
            with redirect_stdout(io.StringIO()):
                resumed=train(continuation)
            complete=torch.load(Path(full['output_dir'])/'last.pt',map_location='cpu',weights_only=False)
            actual=torch.load(Path(resumed['output_dir'])/'last.pt',map_location='cpu',weights_only=False)
            for key,value in complete['model'].items():
                if isinstance(value,torch.Tensor):
                    torch.testing.assert_close(value,actual['model'][key],atol=0,rtol=0,msg=key)
                else:
                    self.assertEqual(value,actual['model'][key])
            self.assertEqual(actual['training_stage'],'joint')
            self.assertEqual(actual['global_step'],complete['global_step'])
            for key,state in complete['optimizer']['state'].items():
                for name,value in state.items():
                    torch.testing.assert_close(value,actual['optimizer']['state'][key][name],atol=0,rtol=0)

    def test_invalid_backend_contracts_require_explicit_neutral_settings(self):
        invalid=[dict(contrastive_weight=.001),dict(candidate_backend='legacy_question'),
                 dict(learned_residual_enabled=False,residual_fusion='concat'),
                 dict(profile_residual_condition='something'),dict(legacy_residual_mode='both'),
                 dict(learned_residual_enabled=False,profile_residual_condition='detach')]
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                DHAKT(8,4,embedding_dim=8,**kwargs)
        model=DHAKT(8,4,embedding_dim=8,diffusion=dict(time_embedding_dim=8,hidden_dim=16))
        with self.assertRaisesRegex(ValueError,'artifact_path'):
            model.prepare_training(object())


if __name__=='__main__':
    unittest.main()
