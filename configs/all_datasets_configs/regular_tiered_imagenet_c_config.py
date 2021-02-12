from pathlib import Path

from src.datasets import TieredImageNetC

"""
Config for tieredImageNet-C using raw ILSVRC2015 images.
"""

DATASET = TieredImageNetC
DATA_ROOT = Path("/data/etienneb/ILSVRC2015/Data/CLS-LOC/train")
IMAGE_SIZE = 224
SPECS_ROOT = Path("configs/dataset_specs/tiered_imagenet_c")
