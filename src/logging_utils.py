import json
import time
from pathlib import Path


class ExperimentLogger:
    def __init__(self, output_dir, tensorboard=True):
        self.path = Path(output_dir)
        self.stream = (self.path / "metrics.jsonl").open("a", encoding="utf-8")
        self.writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(self.path / "tensorboard"))

    def log(self, values, step):
        row = {"step": step, "time_unix": time.time(), **values}
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        if self.writer:
            for key, value in values.items():
                self.writer.add_scalar(key, value, step)
            self.writer.flush()
        print(f"step={step} " + " ".join(f"{key}={value:.5f}" for key, value in values.items()), flush=True)

    def close(self):
        self.stream.close()
        if self.writer:
            self.writer.close()
