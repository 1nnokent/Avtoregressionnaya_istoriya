"""Compatibility entry point for the original misspelled metrics module."""
import sys
from run_experiment import main

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
