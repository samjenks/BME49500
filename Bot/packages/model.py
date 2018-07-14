import torch
from torch import nn
import numpy as np
from torch import FloatTensor as tensor
from torch.autograd import Variable
import math
import random

"""
Note: I think I will start shallow to get it working, then I might add residuals and dropout layers and go deeper
"""


class ModelOrder(nn.Module):
    """
    output nodes available [a b c]
    a = pass
    b = order
    c = order and alone

    """
    def __init__(self):
        super(ModelOrder, self).__init__()
        self.input = nn.Linear(6, 12)
        self.h1 = nn.Linear(12, 24)
        self.h2 = nn.Linear(24, 10)
        self.out = nn.Linear(10, 3)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(10)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.input(x))
        x = self.ReLu(self.h1(x))
        x = self.ReLu(self.h2(x))

        return self.out(x)


class ModelCall(nn.Module):
    """
    output nodes available [a b c d e]
    a = pass
    b-e are suits: h, d, c, s

    """
    def __init__(self):
        super(ModelCall, self).__init__()
        self.input = nn.Linear(5, 12)
        self.h1 = nn.Linear(12, 24)
        self.h2 = nn.Linear(24, 24)
        self.h3 = nn.Linear(24, 16)
        self.out = nn.Linear(16, 5)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(16)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.input(x))
        x = self.ReLu(self.h1(x))
        x = self.ReLu(self.h2(x))
        x = self.ReLu(self.h3(x))

        return self.out(x)


class ModelStrat(nn.Module):
    def __init__(self):
        super(ModelStrat, self).__init__()
        self.rnn1 = nn.GRU(input_size=10,
                           hidden_size=128,
                           num_layers=1)
        self.dense1 = nn.Linear(128, 5)

    def forward(self, x, hidden):
        x = x.view(1, 1, 10)
        x, hidden = self.rnn1(x, hidden)
        x = self.dense1(x)
        return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(1, batch_size, 128).zero_())


class ModelDiscard(nn.Module):
    """
    output nodes available [a b c d e]
    correspond to idx of cards in hand to replace

    """
    def __init__(self):
        super(ModelDiscard, self).__init__()
        self.input = nn.Linear(7, 12)
        self.h1 = nn.Linear(12, 24)
        self.h2 = nn.Linear(24, 24)
        self.h3 = nn.Linear(24, 16)
        self.out = nn.Linear(16, 5)

        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(16)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.ReLu(self.input(x))
        x = self.ReLu(self.h1(x))
        x = self.ReLu(self.h2(x))
        x = self.ReLu(self.h3(x))

        return self.out(x)

