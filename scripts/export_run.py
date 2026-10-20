"""Publish lightweight evidence; checkpoints and datasets remain in runs/."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="reports/baseline")
    args = parser.parse_args()
    source, destination = Path(args.run_dir), Path(args.output)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "completed":
        raise ValueError("Only completed runs can be exported")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["config.yaml", "metadata.json", "summary.json", "tokenizer.json", "metrics.jsonl",
                 "samples.png", "real.png", "reconstruction.png", "loss.png", "evaluation.json"]:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    print(destination.resolve())


if __name__ == "__main__":
    main()
