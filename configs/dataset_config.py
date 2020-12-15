from pathlib import Path

from src.datasets import *

# Parameters of the dataset

DATASET = CIFAR100CMeta
IMAGE_SIZE = 32
DATA_ROOT = Path("data")
SPECS_ROOT = Path("configs/dataset_specs/cifar_100_c")
