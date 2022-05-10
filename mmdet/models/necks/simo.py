import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.ops import ConvModule
from ..registry import NECKS
from mmdet.ops.scale_grad import scale_tensor_gard

@NECKS.register_module
class SIMO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level,
                 num_outs,
                 num_down,
                 grad_scale=None):
        super(SIMO, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.fp16_enabled = False
        self.grad_scale = grad_scale

        self.encode_layer = nn.Conv2d(in_channels[start_level], out_channels, kernel_size=1, stride=1, padding=0)
        self.deconvs = nn.ModuleList()
        self.num_down = num_down
        self.extra_convs = nn.ModuleList()
        num_extra_convs = num_outs - num_down - 1
        for i in range(num_extra_convs):
            self.extra_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2,
                                              padding=1))
        self.start_level = start_level

        self.out_convs = nn.ModuleList()
        for i in range(num_down+1):
            self.out_convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        feature = inputs[self.start_level]

        if (self.grad_scale is not None) and self.training:
            feature = scale_tensor_gard(feature, self.grad_scale)

        en_feat = self.encode_layer(feature)
        laterals = [en_feat]

        for i in range(self.num_down):
            laterals = [F.interpolate(laterals[0], size=inputs[self.start_level-1-i].shape[2:], mode='nearest')] + laterals

        outs = []
        for i, out_conv in enumerate(self.out_convs):
            outs.append(out_conv(laterals[i]))

        for i, extra_conv in enumerate(self.extra_convs):
            in_feat = laterals[-1] if i == 0 else outs[-1]
            outs.append(extra_conv(in_feat))

        return tuple(outs)
