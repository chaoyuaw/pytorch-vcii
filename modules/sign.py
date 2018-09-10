import torch
import torch.nn as nn

from functions import Sign as SignFunction


class Sign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SignFunction.apply(x, self.training)
