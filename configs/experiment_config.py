from pathlib import Path

# Global config for the experiment

RANDOM_SEED = 1  # If None, random seed will be randomly sampled
SAVE_RESULTS = True
OVERWRITE = False  # If True, will erase all previous content of SAVE_DIR
SAVE_DIR = Path("output") / "results"
