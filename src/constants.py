from enum import Enum
from pathlib import Path

import torch

from src.methods import backbones

SAVE_DIR = Path("output")
SPECS_DIR = Path("data") / "specs"


class SplitKeys(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Datasets(Enum):
    AIRCRAFT = "aircraft"
    CU_BIRDS = "cu_birds"
    DTD = "dtd"
    FUNGI = "fungi"
    ILSVRC_2012 = "ilsvrc_2012"
    MSCOCO = "mscoco"
    OMNIGLOT = "omniglot"
    QUICKDRAW = "quickdraw"
    TRAFFIC_SIGN = "traffic_sign"
    VGG_FLOWER = "vgg_flower"


backbones_dict = dict(
    Conv4=backbones.Conv4,
    Conv4S=backbones.Conv4S,
    Conv6=backbones.Conv6,
    ResNet10=backbones.ResNet10,
    ResNet18=backbones.ResNet18,
    ResNet34=backbones.ResNet34,
    ResNet50=backbones.ResNet50,
    ResNet101=backbones.ResNet101,
)
