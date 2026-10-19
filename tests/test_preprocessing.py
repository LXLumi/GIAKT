"""Event identity, chronological recovery and target-coverage checks."""
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
from scripts.preprocessing import build_index, pack_sequences, split_students
from scripts.assistment17.preprocess import read_clean, recover_chronology, preprocess
from src.utils.config import load_config


class PreprocessingTests(unittest.TestCase):
    def test_indices_preserve_complete_concept_names(self):
        mapping = build_index(["application: compare points", "a-b", "a-b"])
        self.assertEqual(set(mapping), {"application: compare points", "a-b"})
        self.assertEqual(set(mapping.values()), {1, 2})
        self.assertEqual(build_index(["20", "3", "10"]), {"3": 1, "10": 2, "20": 3})

    def test_student_splits_are_disjoint_and_repeatable(self):
        first = split_students(range(1, 101), [.64, .16, .20], 42)
        second = split_students(range(1, 101), [.64, .16, .20], 42)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual([len(x) for x in first.values()], [64, 16, 20])
        self.assertFalse(set(first["train"]) & set(first["valid"]))
        self.assertEqual(len(set(np.concatenate(list(first.values())))), 100)

    def test_windows_preserve_every_target_once_including_two_event_tail(self):
        for length in [3, 4, 5, 6, 7, 8, 9]:
            frame = pd.DataFrame({"studentId": 1, "problemId": np.arange(1, length+1),
                "skill": 1, "correct": 0, "timeTaken": 2., "sourceRow": np.arange(length)})
            packed = pack_sequences(frame, 4, 300)
            targets = [int(value) for row, size in zip(packed["sourceRow"], packed["lengths"])
                       for value in row[1:size]]
            self.assertEqual(targets, list(range(1, length)))
            for row, size in zip(packed["problemId"], packed["lengths"]):
                self.assertTrue((row[size:] == 0).all())

    def test_repeated_attempts_recover_distinct_real_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            clean = pd.DataFrame({"studentId": [1, 1], "problemId": [2, 2], "skill": ["a-b", "a-b"],
                                  "timeTaken": [2., 2.], "correct": [0, 0]})
            raw = clean.assign(startTime=[20, 10], action_num=[7, 8])
            clean.to_csv(path / "clean.csv", index=False)
            raw.to_csv(path / "raw.csv", index=False)
            read, _ = read_clean(path / "clean.csv")
            recovered = recover_chronology(read, path / "raw.csv")
            self.assertEqual(recovered.action_num.tolist(), [7, 8])
            self.assertEqual(recovered.sort_values("startTime").action_num.tolist(), [8, 7])
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                recover_chronology(read.iloc[:1], path / "raw.csv")

    def test_invalid_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.csv"
            pd.DataFrame({"studentId": [1]*5, "problemId": [2]*5, "skill": ["a", "noskill", "a", "a", "a"],
                          "timeTaken": [2, 2, -1, "inf", 2], "correct": [1, 1, 0, 0, .5]}).to_csv(path, index=False)
            frame, removed = read_clean(path)
            self.assertEqual((len(frame), removed), (1, 4))

    def test_end_to_end_bounds_and_existing_event_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"studentId": student, "problemId": step+1, "skill": "a-b",
                     "timeTaken": [2, 2, 400, 4][step], "correct": step % 2,
                     "startTime": step+10, "action_num": student*10+step}
                    for student in range(1, 11) for step in range(4)]
            rows[0]["timeTaken"] = .5
            rows.append(dict(rows[5]))  # Exactly the same event, not another practice.
            clean = root / "clean.csv"
            pd.DataFrame(rows).to_csv(clean, index=False)
            config = load_config()
            config["dataset"]["paths"].update(clean=str(clean), processed=str(root/"processed"),
                                                raw=str(root/"does_not_exist.csv"))
            config["preprocess"]["max_seq_len"] = 4
            with redirect_stdout(io.StringIO()):
                report = preprocess(config)
            self.assertEqual(report["duplicate_events_removed"], 1)
            self.assertEqual(report["below_min_time"], 1)
            self.assertEqual(report["above_max_time"], 10)
            self.assertEqual(report["usable_rows"], 39)
            indexed = pd.read_csv(root/"processed/indexed.csv")
            self.assertEqual(indexed.timeTaken.max(), 300)
            self.assertGreaterEqual(indexed.timeTaken.min(), 1)
            self.assertEqual(sum(x["targets"] for x in report["splits"].values()), 29)


if __name__ == "__main__":
    unittest.main()
