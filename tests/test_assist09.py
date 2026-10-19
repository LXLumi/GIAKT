"""ASSIST09 format adaptation, event identity and atomic-skill model compatibility."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import torch
from scripts.assistment09.preprocess import preprocess, read_raw, clean_events, inspect_raw
from src.data.interactions import InteractionDataset, collate_sequences
from scripts.preprocessing import FIELDS
from src.model.diffuqkt import DiffuQKT
from src.utils.config import load_config
from src.utils.files import sha256
from test_diffuqkt import example_batch


def raw_event(student, order, milliseconds=2000, skill="1_13"):
    return dict(user_id=student, problem_id=100, skill_id=skill,
                ms_first_response=milliseconds, correct=order % 2, order_id=order,
                skill_name="", answer_text="unused text", hint_count=9)


class Assist09PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/assistment2009.yaml")

    def test_time_units_boundary_and_missing_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            times = [0, -1, 999, 1000, 1235, 300001, 2000, 2000]
            rows = [raw_event(1, i, time) for i, time in enumerate(times)]
            rows[-2]["skill_id"] = ""
            rows[-1]["correct"] = 2
            pd.DataFrame(rows).to_csv(path, index=False)
            frame, report = read_raw(path)
            cleaned = clean_events(frame, self.config["preprocess"], report)
            self.assertEqual(report["invalid_rows_removed"], 4)
            self.assertEqual(report["below_min_time"], 1)
            self.assertEqual(report["above_max_time"], 1)
            self.assertEqual(cleaned.timeTaken.tolist(), [1., 1.24, 300.])
            self.assertEqual(cleaned.rawSkills.tolist(), [("1", "13")] * 3)
            self.config["preprocess"]["short_time_policy"] = "clip"
            clipped = clean_events(frame, self.config["preprocess"], {})
            self.assertEqual(clipped.timeTaken.tolist(), [1., 1., 1.24, 300.])

    def test_event_identity_keeps_distinct_repeated_answers_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            a, b = raw_event(1, 2), raw_event(1, 4)
            pd.DataFrame([a, dict(a), b]).to_csv(path, index=False)
            frame, report = read_raw(path)
            kept = clean_events(frame, self.config["preprocess"], report)
            self.assertEqual(kept.orderId.tolist(), [2, 4])
            self.assertEqual(report["duplicate_events_removed"], 1)
            frame.loc[frame.index[1], "correct"] = 1
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                clean_events(frame, self.config["preprocess"], {})

    def test_end_to_end_five_fields_chronology_atomic_ids_and_8_1_1_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [raw_event(student, student * 100 + step, skill="13_1_1" if step % 2 else "2")
                    for student in range(1, 11) for step in range(7)]
            np.random.default_rng(19).shuffle(rows)
            raw = root / "raw.csv"
            pd.DataFrame(rows).to_csv(raw, index=False)
            before = sha256(raw)
            self.config["dataset"]["paths"].update(raw=str(raw), clean=str(root/"clean.csv"),
                                                    processed=str(root/"processed"))
            self.config["preprocess"]["max_seq_len"] = 4
            # This fixture tests 8:1:1 explicitly, independent of experiment defaults.
            self.config["preprocess"]["split_ratios"] = [.8, .1, .1]
            with redirect_stdout(io.StringIO()):
                report = preprocess(self.config)
                inspection = inspect_raw(self.config, rows=5)
            self.assertEqual(sha256(raw), before)
            self.assertEqual(inspection["sample_rows"], 5)
            self.assertEqual(report["num_skills"], 3)  # atomic 1, 2, 13, sharing 1 across sets
            self.assertEqual([x["students"] for x in report["splits"].values()], [8, 1, 1])
            self.assertEqual(report["usable_rows"], 70)
            self.assertEqual(sum(x["targets"] for x in report["splits"].values()), 60)
            self.assertEqual(report["multi_skill_rows"], 30)
            for csv in [root/"clean.csv", root/"processed/indexed.csv"]:
                self.assertEqual(pd.read_csv(csv).columns.tolist(), FIELDS)
            mappings = json.loads((root/"processed/mappings.json").read_text())
            self.assertEqual(mappings["skill"]["raw_to_index"], {"1": 1, "2": 2, "13": 3})
            seen_students, seen_targets = set(), set()
            for name in ["train", "valid", "test"]:
                with np.load(root/f"processed/{name}.npz", allow_pickle=False) as packed:
                    self.assertEqual(packed["skill"].shape[-1], 2)
                    students = set(packed["studentId"])
                    self.assertFalse(seen_students & students)
                    seen_students |= students
                    for ids, skills, length in zip(packed["orderId"], packed["skill"], packed["lengths"]):
                        self.assertTrue((np.diff(ids[:length]) > 0).all())
                        self.assertTrue((skills[length:] == 0).all())
                        for event in ids[1:length]:
                            self.assertNotIn(int(event), seen_targets)
                            seen_targets.add(int(event))
                limited = InteractionDataset(root/f"processed/{name}.npz", max_seq_len=3)
                batch = collate_sequences([limited[0]])
                self.assertEqual(batch["skill_ids"].shape, (1, 3, 2))
            self.assertEqual(len(seen_targets), 60)
            relations = pd.read_csv(root/"processed/question_concepts.csv")
            self.assertEqual(set(relations.skill), {1, 2, 3})


class MultiSkillModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.model = DiffuQKT(8, 4, embedding_dim=8, num_heads=2, dropout=0, diffusion_steps=4)

    def predict(self, batch):
        self.model.eval()
        with torch.no_grad():
            return self.model(batch, generator=torch.Generator().manual_seed(19))

    def test_single_skill_predictions_and_checkpoint_structure_are_preserved(self):
        batch = example_batch()
        expected = self.predict(batch)
        # Additional skill slots containing padding must not scale a singleton embedding.
        batch["skill_ids"] = torch.stack([batch["skill_ids"], torch.zeros_like(batch["skill_ids"])], dim=-1)
        actual = self.predict(batch)
        torch.testing.assert_close(actual["probabilities"], expected["probabilities"], atol=0, rtol=0)
        torch.testing.assert_close(actual["kt_loss"], expected["kt_loss"], atol=0, rtol=0)

    def test_skill_order_padding_and_shared_atom_gradients(self):
        ids = torch.tensor([[[1, 3, 0], [3, 0, 0], [0, 0, 0]]])
        expected = torch.stack([(self.model.concept.weight[1] + self.model.concept.weight[3])/2,
                                self.model.concept.weight[3], torch.zeros(8)])[None]
        actual = self.model.embed_skills(self.model.concept, ids)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual, self.model.embed_skills(self.model.concept, ids.flip(-1)))
        actual.sum().backward()
        torch.testing.assert_close(self.model.concept.weight.grad[1], torch.full((8,), .5))
        torch.testing.assert_close(self.model.concept.weight.grad[3], torch.full((8,), 1.5))
        self.assertEqual(float(self.model.concept.weight.grad[0].abs().sum()), 0.)

    def test_joint_objective_multiskill_gradients_and_temporal_causality(self):
        batch = example_batch()
        extra = torch.where(batch["valid_mask"], torch.tensor(4), torch.tensor(0))
        batch["skill_ids"] = torch.stack([batch["skill_ids"], extra], dim=-1)
        self.model.train()
        result = self.model(batch)
        self.assertGreater(float(result["contrastive_loss"]), 0.)
        self.assertGreater(float(result["diffusion_loss"]), 0.)
        result["loss"].backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(self.model.concept.weight.grad[4].abs().sum()), 0.)
        self.assertGreater(float(self.model.variation.weight.grad[4].abs().sum()), 0.)
        before = self.predict(batch)["probabilities"]
        batch["answers"][:, 2:] = 1 - batch["answers"][:, 2:]
        after = self.predict(batch)["probabilities"]
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
