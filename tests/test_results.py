"""Best-epoch reporting, legacy summaries and CSV grouping."""
import ast
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import unittest

from src.training.results import experiment_rows, format_duration, rebuild_results_csv


class ExperimentResultsTests(unittest.TestCase):
    def write_summary(self, directory, **changes):
        directory.mkdir(parents=True, exist_ok=True)
        summary = {"best_epoch": 2, "best_validation": {
            "epoch": 2, "train_auc": .81, "train_accuracy": .72,
            "train_loss": .51, "train_kt_loss": .50,
            "train_contrastive_loss": 2.0, "train_diffusion_loss": .8,
            "valid_auc": .77, "valid_accuracy": .70, "valid_loss": .55},
            "test": {"auc": .76, "accuracy": .69, "loss": .56}}
        summary.update(changes)
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def test_three_rows_keep_name_finish_time_and_duration_in_experiment_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "assistment2017_100epochs_1"
            self.write_summary(run, duration_seconds=3665.2,
                               finished_at="2026-10-18T15:30:45+08:00")
            rows = experiment_rows(run)
            self.assertEqual([row["split"] for row in rows], ["train", "valid", "test"])
            self.assertEqual([row["experiment"] for row in rows],
                             [run.name, "2026-10-18 15:30:45", "61m5s"])
            self.assertEqual([row["备注"] for row in rows], ["", "", ""])
            self.assertNotIn("time", rows[0])
            self.assertEqual(ast.literal_eval(rows[0]["loss"]), (.50, 2., .8))
            self.assertEqual(float(rows[1]["loss"]), .55)
            self.assertEqual(float(rows[2]["loss"]), .56)
            self.assertEqual(float(rows[1]["AUC"]), .77)

    def test_legacy_summary_keeps_metrics_but_does_not_invent_total_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "old_run"
            self.write_summary(run)
            modified = datetime(2026, 10, 16, 16, 10, 12).timestamp()
            os.utime(run / "summary.json", (modified, modified))
            rows = experiment_rows(run)
            self.assertEqual(float(rows[0]["AUC"]), .81)
            self.assertEqual(rows[1]["experiment"], "2026-10-16 16:10:12")
            self.assertEqual(rows[2]["experiment"], "")

    def test_history_fallback_uses_best_epoch_not_last_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.write_summary(run, best_validation={"epoch": 3, "train_auc": .99})
            with (run / "history.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["epoch", "train_auc", "train_kt_loss",
                    "train_contrastive_loss", "train_diffusion_loss", "valid_auc", "valid_loss"])
                writer.writeheader()
                for epoch in (1, 2, 3):
                    writer.writerow({"epoch": epoch, "train_auc": epoch / 10,
                                     "train_kt_loss": epoch / 20, "train_contrastive_loss": epoch,
                                     "train_diffusion_loss": epoch / 30, "valid_auc": epoch / 5,
                                     "valid_loss": epoch / 4})
            rows = experiment_rows(run)
            self.assertEqual(float(rows[0]["AUC"]), .2)
            self.assertEqual(float(rows[1]["AUC"]), .4)
            self.assertEqual(float(rows[1]["loss"]), .5)

    def test_rebuild_deduplicates_updates_and_skips_unfinished_or_invalid_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "run_1", root / "run_2"
            self.write_summary(first)
            self.write_summary(second)
            (root / "unfinished").mkdir()
            (root / "invalid").mkdir()
            (root / "invalid" / "summary.json").write_text("{", encoding="utf-8")
            with self.assertWarnsRegex(UserWarning, "Skipping invalid"):
                destination = rebuild_results_csv(root)
            self.write_summary(first, test={"auc": .9, "accuracy": .8, "loss": .4})
            with self.assertWarnsRegex(UserWarning, "Skipping invalid"):
                self.assertEqual(rebuild_results_csv(root), destination)
            self.assertTrue(destination.read_bytes().startswith(b"\xef\xbb\xbf"))
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["experiment"] for row in rows if row["split"] == "train"],
                             ["run_1", "run_2"])
            self.assertEqual(float(rows[2]["AUC"]), .9)
            self.assertEqual(list(root.glob(".results_*.tmp")), [])

    def test_rebuild_preserves_all_notes_when_an_earlier_run_is_added(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_summary(root / "run_b", duration_seconds=61,
                               finished_at="2026-10-18T15:30:45+08:00")
            self.write_summary(root / "run_c", duration_seconds=122)
            destination = rebuild_results_csv(root)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns, rows = reader.fieldnames, list(reader)
            notes = ['学习率 0.001, dropout="0.2"', "验证备注\n第二行", "测试备注",
                     "另一实验", "", "保留末行备注"]
            for row, note in zip(rows, notes):
                row["备注"] = note
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            self.write_summary(root / "run_a")
            self.write_summary(root / "run_b", duration_seconds=61,
                               finished_at="2026-10-18T15:30:45+08:00",
                               test={"auc": .9, "accuracy": .8, "loss": .4})
            rebuild_results_csv(root)
            first_rebuild = destination.read_bytes()
            rebuild_results_csv(root)
            self.assertEqual(destination.read_bytes(), first_rebuild)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual([row["备注"] for row in rows[:3]], ["", "", ""])
            self.assertEqual([row["备注"] for row in rows[3:]], notes)
            self.assertEqual(float(rows[5]["AUC"]), .9)
            self.assertEqual(rows[4]["experiment"], "2026-10-18 15:30:45")
            self.assertEqual(rows[5]["experiment"], "1m1s")

    def test_rebuild_migrates_old_time_column_to_blank_notes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "old_run"
            self.write_summary(run, duration_seconds=3665.2)
            destination = root / "results.csv"
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=
                                        ["experiment", "split", "AUC", "ACC", "loss", "time"])
                writer.writeheader()
                writer.writerows([{"experiment": run.name if split == "train" else "",
                                   "split": split, "time": "61m5s" if split == "train" else ""}
                                  for split in ("train", "valid", "test")])
            rebuild_results_csv(root)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames,
                                 ["experiment", "split", "AUC", "ACC", "loss", "备注"])
                rows = list(reader)
            self.assertEqual(rows[2]["experiment"], "61m5s")
            self.assertTrue(all(row["备注"] == "" for row in rows))

    def test_grouped_and_flat_runs_share_one_report_and_notes_survive_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "assistment2017_50epochs_1"
            self.write_summary(legacy)
            self.write_summary(root / "linearq" / "assistment2009" / "linearq_assistment2009_50epochs_1",
                               results_decimal_places=4)
            self.write_summary(root / "linearq" / "assistment2017" / "linearq_assistment2017_50epochs_1",
                               results_decimal_places=4)
            destination = rebuild_results_csv(root)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns, before = reader.fieldnames, list(reader)
            self.assertEqual(len(before), 9)
            self.assertEqual([row["experiment"] for row in before if row["split"] == "train"],
                             [legacy.name, "linearq_assistment2009_50epochs_1",
                              "linearq_assistment2017_50epochs_1"])
            before[1]["备注"] = "验证备注, 迁移后保留"
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(before)
            original_summary = (legacy / "summary.json").read_bytes()
            group = root / "diffuqk" / "assistment2017"
            group.mkdir(parents=True)
            moved = legacy.rename(group / legacy.name)
            rebuild_results_csv(root)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                after = list(csv.DictReader(stream))
            self.assertEqual(after, before)
            self.assertEqual((moved / "summary.json").read_bytes(), original_summary)
            self.assertEqual(list(root.rglob("results.csv")), [destination])

    def test_cached_history_without_run_artifacts_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "archived_run"
            self.write_summary(run)
            destination = rebuild_results_csv(root)
            original = destination.read_bytes()
            (run / "summary.json").unlink()
            rebuild_results_csv(root)
            self.assertEqual(destination.read_bytes(), original)
            self.write_summary(root / "new_run", results_decimal_places=4)
            rebuild_results_csv(root)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["experiment"] for row in rows if row["split"] == "train"],
                             ["archived_run", "new_run"])

    def test_missing_auxiliary_losses_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self.write_summary(run, train={"auc": None, "contrastive_loss": None,
                                           "diffusion_loss": None})
            row = experiment_rows(run)[0]
            self.assertEqual(row["AUC"], "")
            self.assertEqual(row["loss"], "(0.50000000, NA, NA)")

    def test_new_runs_round_report_only_to_four_decimals(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "new_run"
            self.write_summary(run, results_decimal_places=4,
                train={"auc": .8123456789, "accuracy": .7234567891,
                       "kt_loss": .5012345678, "contrastive_loss": 2.0123456789,
                       "diffusion_loss": .0000123456},
                validation={"auc": .7712345678, "accuracy": .7, "loss": .5512345678},
                test={"auc": None, "accuracy": .6912345678, "loss": .5612345678})
            original = (run / "summary.json").read_bytes()
            rows = experiment_rows(run)
            self.assertEqual(rows[0]["AUC"], "0.8123")
            self.assertEqual(rows[0]["ACC"], "0.7235")
            self.assertEqual(rows[0]["loss"], "(0.5012, 2.0123, 0.0000)")
            self.assertEqual(rows[1]["AUC"], "0.7712")
            self.assertEqual(rows[1]["ACC"], "0.7000")
            self.assertEqual(rows[1]["loss"], "0.5512")
            self.assertEqual(rows[2]["AUC"], "")
            self.assertEqual(rows[2]["ACC"], "0.6912")
            self.assertEqual(rows[2]["loss"], "0.5612")
            self.assertEqual((run / "summary.json").read_bytes(), original)

    def test_new_runs_keep_legacy_csv_values_and_notes_during_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_summary(root / "legacy_run",
                               train={"auc": .8123456789, "kt_loss": .5012345678})
            destination = rebuild_results_csv(root)
            with destination.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns, legacy = reader.fieldnames, list(reader)
            legacy[0]["备注"] = "保留旧实验备注"
            with destination.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(legacy)
            self.write_summary(root / "new_run", results_decimal_places=4,
                               train={"auc": .8123456789, "kt_loss": .5012345678})
            for _ in range(2):
                rebuild_results_csv(root)
                with destination.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(rows[:3], legacy)
                self.assertEqual(rows[0]["AUC"], "0.81234568")
                self.assertEqual(rows[3]["AUC"], "0.8123")
                self.assertEqual(rows[3]["loss"], "(0.5012, 2.0000, 0.8000)")

    def test_duration_rounding_minutes_and_invalid_values(self):
        for seconds, expected in [(None, ""), (0, "0m0s"), (.5, "0m1s"),
                                  (59.6, "1m0s"), (3599.6, "60m0s"), (7265, "121m5s")]:
            with self.subTest(seconds=seconds):
                self.assertEqual(format_duration(seconds), expected)
        for seconds in (-1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                format_duration(seconds)


if __name__ == "__main__":
    unittest.main()
