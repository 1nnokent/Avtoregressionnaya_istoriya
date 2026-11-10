import json
import re

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
