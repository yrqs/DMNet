import torch.nn as nn

from mmdet.core import auto_fp16
from ..registry import NECKS
from .defpn import DeFPN

@NECKS.register_module
class DDeFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 num_outs,
                 num_deconv,
                 grad_scale=None):
        assert isinstance(start_level, list)
        assert isinstance(num_deconv, list)
        super(DDeFPN, self).__init__()
        self.branch1 = DeFPN(in_channels, out_channels, start_level[0], num_outs, num_deconv[0], grad_scale)
        self.branch2 = DeFPN(in_channels, out_channels, start_level[1], num_outs, num_deconv[1], grad_scale)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        self.branch1.init_weights()
        self.branch2.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        outs1 = self.branch1(inputs)
        outs2 = self.branch2(inputs)

        outs = [o for o in zip(outs1, outs2)]

        return tuple(outs)
