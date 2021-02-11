from functools import partial
from pathlib import Path

from src.datasets import *

# Parameters of the dataset
# DATASET = CIFAR100CMeta
# IMAGE_SIZE = 32
# DATA_ROOT = Path("./data")
# SPECS_ROOT = Path("configs/dataset_specs/cifar_100_c")

# DATASET = TieredImageNetC
# DATA_ROOT = Path("/data/etienneb/ILSVRC2015/Data/CLS-LOC/train")
DATASET = partial(TieredImageNetC, load_corrupted_dataset=True)
DATA_ROOT = Path("/media/etienneb/tiered_imagenet_c/")

IMAGE_SIZE = 224
SPECS_ROOT = Path("configs/dataset_specs/tiered_imagenet_c")

