from torch import nn as nn
import math
from src.modules.backbones import *

from src.methods.utils import softplus
from src.modules.backbones import ConvBlock, init_layer



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

        self.num_features = num_features

        if self.num_features not in [256, 1600]:
            raise NotImplementedError('RelationNet is not implemented for num_features='+str(num_features))

        if self.num_features == 256:
            self.conv_1 = ConvBlock(64, 1, pool=True)
            self.conv = self.conv_1
        elif self.num_features == 1600:
            self.conv_1 = ConvBlock(64, 64, pool=True)
            self.conv_2 = ConvBlock(64, 1, pool=True)
            self.conv = nn.Sequential(self.conv_1, self.conv_2)

        self.lin_1 = nn.Linear(1, 8)
        self.lin_2 = nn.Linear(8, 1)

        # We enforce to start from sigma=1. to prevent learning failure. MANDATORY!
        self.lin_2.weight.data.fill_(0.)
        self.lin_2.bias.data.fill_(1.)

        flatten = nn.Flatten()
        relu = nn.ReLU()

        self.net = nn.Sequential(
            flatten, 
            self.lin_1, 
            relu, 
            self.lin_2
        )

    def forward(self, x):
        if self.num_features == 256:
            x = x.reshape(-1, 64, 2, 2)
        elif self.num_features == 1600:
            x = x.reshape(-1, 64, 5, 5)
        x = self.conv_1(x)
        sigma = self.net(x)
        return sigma


