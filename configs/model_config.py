from functools import partial

from src.modules.batch_norm import *

BATCHNORM = ConventionalBatchNorm

from src.modules.backbones import *
from src.modules import *
from src.methods import *

# Parameters of the model (method and feature extractor)

BACKBONE = ResNet18

TRANSPORTATION_MODULE = OptimalTransport(
    regularization=0.05,
    max_iter=1000,
    stopping_criterion=1e-4,
)

MODEL = partial(
    ProtoNet,
    transportation=TRANSPORTATION_MODULE,
)
