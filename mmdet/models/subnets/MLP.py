import torch.nn as nn

from mmcv.cnn import normal_init

class MLP(nn.Module):
    def __init__(self,
                 num_layer,
                 in_channels,
                 out_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(nn.Linear(in_channels, out_channels))
            if i < num_layer-1:
                self.layers.append(nn.ReLU())

        for m in self.layers:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
