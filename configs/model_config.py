from functools import partial

from src.methods.backbones import *
from src.methods import *

# Parameters of the model (method and feature extractor)

# MODEL = ProtoNet
MODEL = partial(OptimalTransport, regularization=0.05, max_iter=1000)
BACKBONE = Conv4
