"""Compatibility launcher; all configuration is owned by run_experiment.py."""
import sys
from run_experiment import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    main()
