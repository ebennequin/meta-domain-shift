from functools import partial

from src.methods.backbones import *
from src.methods import *

# Parameters of the model (method and feature extractor)

# MODEL = ProtoNet
MODEL = partial(OptimalTransport, regularization=0.1, max_iter=100)
BACKBONE = ResNet50
