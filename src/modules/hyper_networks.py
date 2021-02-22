import numpy as np
from torch import nn as nn
import math
from src.modules.backbones import *

from src.methods.utils import softplus


class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_features):
        super(MultiLayerPerceptron, self).__init__()

        self.layer_1 = nn.Linear(num_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.layer_2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.layer_3 = nn.Linear(1024, 1)

        relu = nn.ReLU()

        self.mlp = nn.Sequential(
            self.layer_1, self.bn1, relu, self.layer_2, self.bn2, relu, self.layer_3
        )

    def forward(self, x):
        return softplus(self.mlp(x))


class RelationNet(nn.Module):
    """
    RelationNet for Transductive Propagation Net.
    Used for scaling the similarity.
    Uses a ConvNet architecture.
    """

    def __init__(self, num_features):
        super(RelationNet, self).__init__()

        self.lin_1 = nn.Linear(num_features, 64)
        self.lin_2 = nn.Linear(64, 8)
        self.lin_3 = nn.Linear(8, 1)

        # We enforce to start from sigma=0.1 to prevent learning failure. MANDATORY!
        self.lin_1.weight.data.fill_(0.0)
        self.lin_1.bias.data.fill_(1.0)
        self.lin_2.weight.data.fill_(0.0)
        self.lin_2.bias.data.fill_(1.0)
        self.lin_3.weight.data.fill_(0.0)
        self.lin_3.bias.data.fill_(1.0)

        relu = nn.ReLU()

        self.net = nn.Sequential(self.lin_1, relu, self.lin_2, relu, self.lin_3)

    def forward(self, x):
        sigma = self.net(x)
        return sigma
