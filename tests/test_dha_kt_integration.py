"""Public entry point, readonly diagnosis, cache contexts and full parameter accounting."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch
import yaml

from src.model import DHAKT,DiffuQKT
from src.model.registry import build_model
from src.model.dha_kt.diagnostics import diagnose_checkpoint
from src.training.cli import main
from src.training.trainer import train
from dha_kt_helpers import make_data,prepared_model


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_cli_selects_independent_config_and_reference_with_smoke(self):
        captured=[]
        with patch('src.training.trainer.train',side_effect=lambda config,overwrite:captured.append(config)):
            main(['assistment2009','--model','dha_kt','--reference-artifact','data/example/reference','--smoke','--epochs','3'])
        config=captured[0]
        self.assertEqual(config['model_name'],'dha_kt')
        self.assertEqual(config['model']['profile']['artifact_path'],'data/example/reference')
        self.assertEqual(config['model']['embedding_dim'],32)
        self.assertEqual(config['model']['diffusion']['sampling_steps'],8)
        self.assertEqual(config['model']['stages']['base_warmup_epochs'],0)
        self.assertEqual(config['loss']['contrastive_weight'],0)
        self.assertNotIn('residual_diffusion_enabled',config['model'])

    def test_cli_without_reference_option_discovers_each_dataset_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            for dataset in ['assistment09','assistment17']:
                with self.subTest(dataset=dataset):
                    root=Path(temporary)/dataset
                    config=make_data(root/'processed')
                    canonical=root/'dha_kt/reference'
                    shutil.copytree(root/'processed/reference',canonical)
                    config['model']['profile']['artifact_path']=None
                    path=root/'config.yaml'
                    path.write_text(yaml.safe_dump(config),encoding='utf-8')
                    with redirect_stdout(io.StringIO()):
                        summary=main([str(path),'--model','dha_kt','--epochs','1'])
                    self.assertEqual(Path(summary['model_kwargs']['profile']['artifact_path']),canonical.resolve())
                    self.assertEqual(summary['model_name'],'dha_kt')

    def test_explicit_reference_wins_and_missing_reference_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=make_data(root/'processed')
            canonical=root/'dha_kt/reference'
            shutil.copytree(root/'processed/reference',canonical)
            from src.data.interactions import InteractionDataset
            from src.training.trainer import model_kwargs
            dataset=InteractionDataset(root/'processed/train.npz')
            metadata=json.loads((root/'processed/metadata.json').read_text(encoding='utf-8'))
            explicit=build_model('dha_kt',model_kwargs(config,metadata))
            explicit.prepare_training(dataset)
            self.assertEqual(Path(explicit.profile['artifact_path']),root/'processed/reference')
            config['model']['profile']['artifact_path']=str(root/'explicit_missing')
            with self.assertRaises(FileNotFoundError):
                build_model('dha_kt',model_kwargs(config,metadata)).prepare_training(dataset)
            config['model']['profile']['artifact_path']=None
            with tempfile.TemporaryDirectory() as other:
                absent=copy.copy(dataset)
                absent.path=Path(other)/'processed/train.npz'
                with self.assertRaisesRegex(FileNotFoundError,'manifest missing'):
                    build_model('dha_kt',model_kwargs(config,metadata)).prepare_training(absent)

    def test_readonly_diagnostics_saved_bank_calibration_and_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            config=make_data(root)
            with redirect_stdout(io.StringIO()):
                summary=train(config)
            path=Path(summary['output_dir'])/'best.pt'
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            for file in (root/'reference').iterdir():
                file.unlink()
            report=diagnose_checkpoint(config,path,max_batches=1)
            self.assertEqual(report['model_name'],'dha_kt')
            self.assertEqual(report['training_stage'],'joint')
            self.assertGreater(report['statistics']['candidate_norm']['count'],0)
            self.assertGreater(report['statistics']['correction_semantic_ratio']['count'],0)
            self.assertIn('brier',report['calibration'])
            self.assertIn('ece',report['pilot_calibration'])
            self.assertTrue(report['gradient_probe']['kt_loss']['diffusion']['has_gradient'])
            self.assertIsNone(report['gradient_probe']['diffusion_loss']['profile_to_residual']['norm'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
            json.dumps(report,allow_nan=False)

    def test_default_parameter_ratio_matches_real_baseline_for_all_fusions(self):
        for problems,skills in [(17645,123),(2210,101)]:
            baseline=DiffuQKT(problems,skills)
            for fusion,keep in [('add',False),('concat',False),('concat',True)]:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model=DHAKT(problems,skills,residual_fusion=fusion,residual_concat_keep_2d=keep)
                self.assertEqual(model.baseline_parameter_count,sum(p.numel() for p in baseline.parameters()))
                self.assertEqual(model.parameter_count,sum(p.numel() for p in model.parameters()))
                self.assertLessEqual(model.parameter_ratio,model.max_parameter_ratio)
        with self.assertRaisesRegex(ValueError,'explicit maximum'):
            DHAKT(2210,101,residual_fusion='concat',residual_concat_keep_2d=True,max_parameter_ratio=2.)

    def test_train_eval_mode_does_not_change_reference_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            model,batch,_=prepared_model(temporary)
            expected=model.reference_bank.views_for_students(batch['student_ids'],'train')
            model.eval()
            model.set_data_context('train')
            result=model(batch,return_details=True)
            actual=model.reference_bank.views_for_students(batch['student_ids'],model.data_context)
            torch.testing.assert_close(expected,actual,atol=0,rtol=0)
            model.set_data_context('valid')
            with self.assertRaisesRegex(ValueError,'membership'):
                model(batch)


if __name__=='__main__':
    unittest.main()
