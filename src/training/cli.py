"""Command-line configuration for training and evaluating DiffuQKT."""
import argparse
from pathlib import Path

from src.utils.config import load_config, project_path


DATASET_CONFIGS = {
    "assistment2009": "configs/assistment2009.yaml",
    "assistment2017": "configs/assistment2017.yaml",
}
DATASET_ALIASES = {
    "assistment09": "assistment2009",
    "assistment17": "assistment2017",
    "assistmnet2017": "assistment2017",
}


def resolve_dataset_config(selector):
    """Accept a dataset name or YAML path, defaulting to ASSIST09."""
    selector = selector or "assistment2009"
    canonical = DATASET_ALIASES.get(selector.lower(), selector.lower())
    if canonical in DATASET_CONFIGS:
        return DATASET_CONFIGS[canonical]
    if Path(selector).suffix.lower() in {".yaml", ".yml"}:
        return selector
    raise ValueError("Unknown dataset. Use assistment2009, assistment2017, or a YAML configuration path.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help="Dataset name or YAML path (default: assistment2009)")
    parser.add_argument("--config", help="Explicit YAML configuration path; alternative to the dataset argument")
    parser.add_argument("--preprocess", action="store_true", help="Rebuild canonical data and student splits before training")
    parser.add_argument("--smoke", action="store_true", help="Use two small epochs; explicit options override smoke defaults")
    parser.add_argument("--evaluate", metavar="CHECKPOINT", help="Evaluate saved weights instead of training")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test", help="Evaluation split (default: test)")
    parser.add_argument("--epochs", type=int, help="Maximum epochs (base configuration default: 100)")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--seed", type=int, help="Override training and preprocessing seed")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0 (default: auto)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-root", help="Parent for automatically numbered training directories (default: experiments)")
    output.add_argument("--output-dir", help="Optional exact output directory, retaining the previous command syntax")
    parser.add_argument("--max-windows", type=int, help="Limit windows per split for a small check")
    parser.add_argument("--max-seq-len", type=int, help="Limit sequence length for a small check")
    parser.add_argument("--max-batches", type=int, help="Limit batches per split for a small check")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing a run selected with --output-dir")
    args = parser.parse_args(argv)
    if args.dataset is not None and args.config is not None:
        parser.error("Use either the dataset argument or --config, not both.")
    if args.evaluate and args.preprocess:
        parser.error("--preprocess cannot be combined with --evaluate; evaluation preserves the checkpoint's split.")
    if args.evaluate and args.output_root:
        parser.error("--output-root is for training; use --output-dir to choose an evaluation output directory.")
    try:
        config_path = args.config or resolve_dataset_config(args.dataset)
        config = load_config(config_path)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    control = config["train"]
    if args.smoke:
        config["model"].update(embedding_dim=32, dropout=0.1)
        control.update(epochs=2, batch_size=8, max_windows=64, max_seq_len=32)
    for name in ["epochs", "batch_size", "seed", "device", "max_windows", "max_seq_len", "max_batches"]:
        value = getattr(args, name)
        if value is not None:
            control[name] = value
    if args.output_root is not None:
        control["output_root"] = args.output_root
        control.pop("output_dir", None)
    elif args.output_dir is not None:
        control["output_dir"] = args.output_dir
    if args.preprocess:
        control["preprocess_each_run"] = True
    from src.training.trainer import train, evaluate_checkpoint
    if args.evaluate:
        destination = control.get("output_dir") or project_path(args.evaluate).parent
        return evaluate_checkpoint(config, args.evaluate, args.split, destination)
    return train(config, args.overwrite)
