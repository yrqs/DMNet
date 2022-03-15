import torch.nn as nn

from mmcv.cnn import normal_init

class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 action_function=None):
        super().__init__()
        self.layers = nn.ModuleList()
        if hidden_channels is not None:
            assert isinstance(hidden_channels, tuple)
            for i, hc in enumerate(hidden_channels):
                if i == 0:
                    self.layers.append(nn.Linear(in_channels, hc))
                else:
                    self.layers.append(nn.Linear(hidden_channels[i-1], hc))
                self.layers.append(action_function)
            self.layers.append(nn.Linear(hidden_channels[-1], out_channels))
        for m in self.layers:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
