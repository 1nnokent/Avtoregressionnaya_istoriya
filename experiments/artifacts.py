"""Persist fitted preprocessing, and capture the original trainer's print logs."""
import json
import re
from dataclasses import asdict

import numpy as np

from experiments.config import resolve


def save_tokenizer(path, tokenizer):
    np.savez_compressed(
        path,
        config=json.dumps(asdict(tokenizer.config)),
        image_shape=tokenizer.image_shape,
        offset=tokenizer.normalizer.offset,
        spread=tokenizer.normalizer.spread,
        thresholds=tokenizer.quantizer.thresholds,
        centers=tokenizer.quantizer.centers,
    )


def load_tokenizer(path, target):
    with np.load(path, allow_pickle=False) as state:
        tokenizer = resolve(target)(json.loads(str(state["config"])))
        tokenizer.image_shape = tuple(int(i) for i in state["image_shape"])
        tokenizer.normalizer.offset = state["offset"]
        tokenizer.normalizer.spread = state["spread"]
        tokenizer.quantizer.thresholds = state["thresholds"]
        tokenizer.quantizer.centers = state["centers"]
    return tokenizer


class RunLog:
    def __init__(self, directory, stream):
        self.stream = stream
        self.console = (directory / "console.log").open("w", encoding="utf-8")
        self.events = (directory / "metrics.jsonl").open("w", encoding="utf-8")
        self.pending = ""

    def record(self, event, **values):
        self.events.write(json.dumps({"event": event, **values}, allow_nan=False) + "\n")
        self.events.flush()

    def write(self, message):
        self.stream.write(message)
        self.console.write(message)
        self.console.flush()
        self.pending += message
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            values = {key: float(value) for key, value in re.findall(r"(epoch|step|train_loss|val_loss|val_perplexity)=([0-9.eE+\-]+)", line)}
            if "train_loss" in values or "val_loss" in values:
                self.record("training", **values)
        return len(message)

    def flush(self):
        self.stream.flush()
        self.console.flush()

    def close(self):
        self.console.close()
        self.events.close()
