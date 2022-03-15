import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    A learnable scale parameter
    """

    def __init__(self, scale=1.0, requires_grad=True):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float), requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale
