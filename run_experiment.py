"""python run_experiment.py train --config configs/mnist_baseline.yaml"""
import argparse

from src.config import load_config
from src.experiment import evaluate_run, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Config-driven autoregressive image experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit tokenizer, train, generate, evaluate and save artifacts")
    train.add_argument("--config", default="configs/mnist_baseline.yaml")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--run-id")
    check = commands.add_parser("validate", help="Resolve and validate config without downloading data")
    check.add_argument("--config", default="configs/mnist_baseline.yaml")
    check.add_argument("--set", action="append", default=[])
    evaluate = commands.add_parser("evaluate", help="Evaluate saved best model and tokenizer without fitting")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.add_argument("--device", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    if args.command == "train":
        run_experiment(args.config, args.set, args.run_id)
    elif args.command == "evaluate":
        evaluate_run(args.run_dir, args.device)
    else:
        import yaml
        print(yaml.safe_dump(load_config(args.config, args.set), sort_keys=False))


if __name__ == "__main__":
    main()
