from pathlib import Path

from src.datasets import FEMNIST

DATASET = FEMNIST
IMAGE_SIZE = 28
DATA_ROOT = Path("./data/femnist")
SPECS_ROOT = Path("configs/dataset_specs/femnist")
