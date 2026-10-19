"""Cross-fitting, provenanced overlap removal and immutable profile coordinates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.dha_kt.build_reference import build_reference, extract_events, prefix_predictions
from src.data.interactions import InteractionDataset
from src.model.dha_kt.reference import ReferenceBank
from src.utils.files import sha256
from dha_kt_helpers import make_data, prepared_model


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_prefix_prediction_precedes_response_and_global_updates_once(self):
        events = [(1, 1, 1, 1, (1, 2)), (1, 2, 1, 0, (1, 2)), (1, 3, 1, 1, ())]
        result = prefix_predictions(events)
        self.assertEqual(result[0], .5)
        self.assertAlmostEqual(result[1], (1+5*(2/3))/6)
        self.assertAlmostEqual(result[2], .5)
        changed = [(s, e, q, 1-y, k) for s,e,q,y,k in events]
        self.assertEqual(prefix_predictions(changed)[0], result[0])

    def test_overlap_identity_preserves_genuine_repeated_question_answer_events(self):
        arrays = dict(studentId=np.array([1,1]), lengths=np.array([3,3]),
            problemId=np.ones((2,3), dtype=np.int64), correct=np.ones((2,3), dtype=np.int64),
            skill=np.ones((2,3), dtype=np.int64), sourceRow=np.array([[1,2,3],[3,4,5]]))
        events, duplicates = extract_events(arrays)
        self.assertEqual((len(events), duplicates), (5,1))
        arrays["correct"][1,0] = 0
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            extract_events(arrays)
        arrays.pop("sourceRow")
        with self.assertRaisesRegex(ValueError, "provenance"):
            extract_events(arrays)

    def test_excluded_fold_and_heldout_labels_never_enter_reference_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = make_data(root)
            original = ReferenceBank()
            original.load_artifact(root/"reference")
            excluded = original.manifest["views"][0]
            excluded_students = set(np.where(original.student_fold_ids.numpy() == 0)[0].tolist())
            with np.load(root/"train.npz", allow_pickle=False) as data:
                arrays = {k:data[k].copy() for k in data.files}
            selected = np.isin(arrays["studentId"], list(excluded_students))
            arrays["correct"][selected] = 1-arrays["correct"][selected]
            np.savez_compressed(root/"train.npz", **arrays)
            build_reference(config, root/"changed_train", bins=4, draw_count=3)
            changed = ReferenceBank()
            changed.load_artifact(root/"changed_train")
            for name in ["partial_profile", "partial_counts", "dense_profile", "dense_counts"]:
                torch.testing.assert_close(getattr(original,name)[0], getattr(changed,name)[0], atol=0, rtol=0)
            for split in ["valid", "test"]:
                with np.load(root/f"{split}.npz", allow_pickle=False) as data:
                    arrays = {k:data[k].copy() for k in data.files}
                arrays["correct"] = 1-arrays["correct"]
                np.savez_compressed(root/f"{split}.npz", **arrays)
            build_reference(config, root/"changed_heldout", bins=4, draw_count=3)
            heldout = ReferenceBank()
            heldout.load_artifact(root/"changed_heldout")
            for name in ["partial_profile", "dense_profile", "partial_counts", "dense_counts"]:
                torch.testing.assert_close(getattr(changed,name), getattr(heldout,name), atol=0, rtol=0)
            self.assertFalse(excluded_students & set(excluded["reference_students"]))

    def test_fixed_coordinates_nested_subsets_masks_and_split_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, batch, _ = prepared_model(temporary)
            bank = model.reference_bank
            for view in bank.manifest["views"]:
                for draw in view["draws"]:
                    self.assertLessEqual(set(draw["partial_students"]), set(draw["dense_students"]))
            torch.testing.assert_close(bank.partial_observed, bank.partial_counts > 0)
            self.assertTrue((bank.dense_profile[~bank.dense_observed] == 0).all())
            train_views = bank.views_for_students(batch["student_ids"], "train")
            model.eval()
            torch.testing.assert_close(train_views, bank.views_for_students(batch["student_ids"], "train"))
            with self.assertRaisesRegex(ValueError, "membership"):
                bank.views_for_students(batch["student_ids"], "valid")
            self.assertEqual(int(bank.views_for_students(torch.tensor([7]), "valid")[0]), 3)
            with self.assertRaisesRegex(ValueError, "Unknown student"):
                bank.views_for_students(torch.tensor([99]), "test")

    def test_hash_scope_missing_provenance_and_selfcontained_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, batch, config = prepared_model(root)
            explicit = ReferenceBank()
            explicit.load_artifact(root/"reference", sha256(root/"reference/manifest.json"))
            self.assertEqual(explicit.manifest_sha256, sha256(root/"reference/manifest.json"))
            with self.assertRaisesRegex(ValueError, "scope"):
                model.prepare_training(InteractionDataset(root/"train.npz", max_seq_len=3))
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                ReferenceBank().load_artifact(root/"reference", "bad")
            checkpoint = copy.deepcopy(model.reference_bank.state_dict())
            schema = model.reference_bank.schema()
            for file in (root/"reference").iterdir():
                file.unlink()
            restored = ReferenceBank(schema)
            restored.load_state_dict(checkpoint, strict=True)
            restored.validate()
            self.assertTrue(bool(restored.ready))
            torch.testing.assert_close(restored.partial_profile, model.reference_bank.partial_profile)
            with self.assertRaises(FileNotFoundError):
                ReferenceBank().load_artifact(root/"reference")


if __name__ == "__main__":
    unittest.main()
