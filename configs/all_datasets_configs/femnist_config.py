from pathlib import Path

from src.data_tools.datasets import FEMNIST

DATASET = FEMNIST
IMAGE_SIZE = 28
DATA_ROOT = Path("/media/etienneb/femnist")
SPECS_ROOT = Path("configs/dataset_specs/femnist")
CLASSES = {"train": 42, "val": 10, "test": 10}
