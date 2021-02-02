from pathlib import Path

from src.datasets import *

# Parameters of the dataset

DATASET = CIFAR100CMeta
IMAGE_SIZE = 32
CLASSES = {
    'train': 100, 
    'val': 10, 
    'test': 15
}

DATA_ROOT = Path("./data")
SPECS_ROOT = Path("configs/dataset_specs/cifar_100_c")
# DATASET = TieredImageNetC
# IMAGE_SIZE = 224
# DATA_ROOT = Path("/data/etienneb/ILSVRC2015/Data/CLS-LOC/train")
# SPECS_ROOT = Path("configs/dataset_specs/tiered_imagenet_c")
