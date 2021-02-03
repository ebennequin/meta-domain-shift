from torch import nn as nn

from src.methods.utils import softplus


class MultiLayerPerceptron(nn.Module):
    def __init__(self, num_features, out_dim=1, hidden_dim=1024):
        super(MultiLayerPerceptron, self).__init__()

        self.layer_1 = nn.Linear(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, out_dim)

        relu = nn.ReLU()

        self.mlp = nn.Sequential(
            self.layer_1, self.bn1, relu, self.layer_2, self.bn2, relu, self.layer_3
        )

    def forward(self, x):
        return self.mlp(x)
