from torch import nn as nn

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
