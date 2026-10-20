"""From the project root: python -m metrics.src.evalute --checkpoint <path>."""
from experiments.run import main

if __name__ == "__main__":
    main(default_mode="evaluate")
