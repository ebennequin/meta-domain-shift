from functools import partial

from src.methods.backbones import *
from src.methods import *

# Parameters of the model (method and feature extractor)

BACKBONE = ResNet34
# BACKBONE = Conv4

# MODEL = ProtoNet
MODEL = partial(
    PrototypicalOptimalTransport,
    # OptimalTransport,
    regularization=0.05,
    max_iter=1000,
    stopping_criterion=1e-4,
    lambda_cost=0.0,
)
