import torch.nn as nn

from mmdet.core import auto_fp16
from ..registry import NECKS

from .fpn import FPN

@NECKS.register_module
class DFPN3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 grad_scale=None):
        super(DFPN3, self).__init__()
        args_dict = locals()
        args_dict.pop('self')
        args_dict.pop('__class__')
        self.fpn_cls = FPN(**args_dict)
        self.fpn_reg = FPN(**args_dict)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        self.fpn_cls.init_weights()
        self.fpn_reg.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        fpn_cls_outs = self.fpn_cls(inputs)
        fpn_reg_outs = self.fpn_reg(inputs)
        outs = [i for i in zip(fpn_cls_outs, fpn_reg_outs)]
        return tuple(outs)
