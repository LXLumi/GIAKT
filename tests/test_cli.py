"""Dataset defaults, optional CLI overrides and unique experiment directories."""
import io
from contextlib import redirect_stderr
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training.cli import main
from src.training.trainer import allocate_run_directory


class CommandLineTests(unittest.TestCase):
    def training_config(self, arguments):
        with patch("src.training.trainer.train", return_value="trained") as run:
            self.assertEqual(main(arguments), "trained")
        run.assert_called_once()
        return run.call_args[0][0]

    def test_no_arguments_uses_assist09_and_100_epochs(self):
        config = self.training_config([])
        self.assertEqual(config["dataset"]["name"], "assistment2009")
        self.assertEqual(config["train"]["epochs"], 100)
        self.assertEqual(config["train"]["output_root"], "experiments")
        self.assertNotIn("output_dir", config["train"])
        self.assertEqual(config["train"]["device"], "auto")

    def test_dataset_argument_and_epoch_override(self):
        config = self.training_config(["assistment2017", "--epochs", "50"])
        self.assertEqual(config["dataset"]["name"], "assistment2017")
        self.assertEqual(config["train"]["epochs"], 50)
        self.assertNotIn("output_dir", config["train"])

    def test_linearq_selector_uses_only_its_model_defaults(self):
        config = self.training_config(["assistment2017", "--model", "LinearQ", "--epochs", "50"])
        self.assertEqual(config["model_name"], "linearq")
        self.assertNotIn("diffusion_steps", config["model"])
        self.assertGreaterEqual(config["loss"]["diffusion_weight"], 0)
        self.assertIn("question_embedding_dim", config["model"])

    def test_newq_selector_preserves_diffusion_and_adds_only_initial_q_settings(self):
        config = self.training_config(["assistment2009", "--model", "NewQ", "--epochs", "50"])
        self.assertEqual(config["model_name"], "newq")
        self.assertEqual(config["model"]["question_embedding_dim"], 16)
        self.assertEqual(config["model"]["attention_temperature"], 1)
        self.assertEqual(config["model"]["diffusion_steps"], 100)
        self.assertEqual(config["loss"]["diffusion_weight"], .01)
        smoke = self.training_config(["--model", "newq", "--smoke"])
        self.assertEqual(smoke["model"]["embedding_dim"], 32)
        self.assertEqual(smoke["model"]["question_embedding_dim"], 4)

    def test_newq_v2_has_independent_dimensions_and_its_own_smoke_settings(self):
        config = self.training_config(["assistment2009", "--model", "newq_v2"])
        self.assertEqual(config["model_name"], "newq_v2")
        self.assertEqual(config["model"]["attention_embedding_dim"], 8)
        self.assertEqual(config["model"]["residual_embedding_dim"], 8)
        self.assertNotIn("question_embedding_dim", config["model"])
        smoke = self.training_config(["--model", "newq_v2", "--smoke"])
        self.assertEqual(smoke["model"]["embedding_dim"], 32)
        self.assertEqual(smoke["model"]["attention_embedding_dim"], 2)
        self.assertEqual(smoke["model"]["residual_embedding_dim"], 2)
        self.assertEqual(smoke["model"]["diffusion_steps"], 100)

    def test_dataset_aliases(self):
        for alias, expected in [("assistment09", "assistment2009"),
                                ("assistment17", "assistment2017"),
                                ("assistmnet2017", "assistment2017")]:
            with self.subTest(alias=alias):
                self.assertEqual(self.training_config([alias])["dataset"]["name"], expected)

    def test_previous_config_and_exact_output_options_still_work(self):
        config = self.training_config(["--config", "configs/assistment2009.yaml", "--epochs", "50",
                                       "--device", "cuda", "--output-dir", "experiments/manual_run"])
        self.assertEqual(config["train"]["output_dir"], "experiments/manual_run")
        self.assertEqual(config["train"]["device"], "cuda")
        self.assertEqual(config["train"]["epochs"], 50)

    def test_optional_output_root_and_smoke_overrides(self):
        config = self.training_config(["assistment2017", "--smoke", "--epochs", "1",
                                       "--output-root", "experiments/checks"])
        self.assertEqual(config["train"]["output_root"], "experiments/checks")
        self.assertEqual(config["train"]["epochs"], 1)
        self.assertEqual(config["model"]["embedding_dim"], 32)
        self.assertEqual(config["train"]["max_windows"], 64)

    def test_evaluation_defaults_to_checkpoint_directory(self):
        checkpoint = "experiments/diffuqk/assistment2017/assistment2017_50epochs_1/best.pt"
        with patch("src.training.trainer.train") as training, \
             patch("src.training.trainer.evaluate_checkpoint", return_value={"auc": .75}) as evaluate:
            self.assertEqual(main(["assistment2017", "--evaluate", checkpoint]), {"auc": .75})
        training.assert_not_called()
        arguments = evaluate.call_args[0]
        self.assertEqual(arguments[0]["dataset"]["name"], "assistment2017")
        self.assertEqual(arguments[1], checkpoint)
        self.assertEqual(arguments[2], "test")
        self.assertEqual(arguments[3].name, "assistment2017_50epochs_1")

    def test_invalid_or_conflicting_selectors_are_rejected(self):
        for arguments in [["assistment2012"],
                          ["--model", "unknown"],
                          ["assistment2017", "--config", "configs/assistment2009.yaml"],
                          ["--evaluate", "best.pt", "--preprocess"],
                          ["--evaluate", "best.pt", "--output-root", "experiments/other"],
                          ["--output-root", "experiments", "--output-dir", "experiments/manual"]]:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 2)


class ExperimentDirectoryTests(unittest.TestCase):
    def test_linearq_prefix_and_numbering_are_separate_from_original_model(self):
        with tempfile.TemporaryDirectory() as directory:
            first = allocate_run_directory(directory, "assistment2017", 50, "linearq")
            second = allocate_run_directory(directory, "assistment2017", 50, "linearq")
            original = allocate_run_directory(directory, "assistment2017", 50)
            self.assertEqual(first.name, "linearq_assistment2017_50epochs_1")
            self.assertEqual(second.name, "linearq_assistment2017_50epochs_2")
            self.assertEqual(original.name, "assistment2017_50epochs_1")
            self.assertEqual(first.parent, Path(directory) / "linearq" / "assistment2017")
            self.assertEqual(original.parent, Path(directory) / "diffuqk" / "assistment2017")

    def test_registered_models_get_their_own_group_without_directory_code_changes(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("src.utils.config.MODEL_CONFIGS", {"newq": "configs/models/newq.yaml"}):
            run = allocate_run_directory(directory, "assistment2017", 50, "newq")
            self.assertEqual(run.relative_to(directory).as_posix(),
                             "newq/assistment2017/newq_assistment2017_50epochs_1")

    def test_requested_name_and_increment_preserve_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiments"
            first = allocate_run_directory(root, "assistment2017", 50)
            self.assertEqual(first.name, "assistment2017_50epochs_1")
            self.assertEqual(first.parent, root / "diffuqk" / "assistment2017")
            marker = first / "existing_result.txt"
            marker.write_text("keep", encoding="utf-8")
            second = allocate_run_directory(root, "assistment2017", 50)
            self.assertEqual(second.name, "assistment2017_50epochs_2")
            (root / "assistment2017_50epochs_4").mkdir()
            self.assertEqual(allocate_run_directory(root, "assistment2017", 50).name,
                             "assistment2017_50epochs_5")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(allocate_run_directory(root, "assistment2009", 50).name,
                             "assistment2009_50epochs_1")
            self.assertEqual(allocate_run_directory(root, "assistment2017", 100).name,
                             "assistment2017_100epochs_1")

    def test_concurrent_allocations_have_unique_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiments"
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda _: allocate_run_directory(root, "assistment2017", 50), range(8)))
            self.assertEqual(len(set(paths)), 8)
            self.assertEqual({path.name for path in paths},
                             {f"assistment2017_50epochs_{n}" for n in range(1, 9)})

    def test_invalid_name_and_epoch_do_not_create_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiments"
            for dataset, epochs in [("../outside", 50), ("assistment2017", 0)]:
                with self.subTest(dataset=dataset, epochs=epochs), self.assertRaises(ValueError):
                    allocate_run_directory(root, dataset, epochs)
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
