from pathlib import Path

from src.datasets import CIFAR100CMeta

DATASET = CIFAR100CMeta
IMAGE_SIZE = 32
CLASSES = {"train": 100, "val": 10, "test": 15}
DATA_ROOT = Path("./data")
SPECS_ROOT = Path("configs/dataset_specs/cifar_100_c")
