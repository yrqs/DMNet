import torch.nn as nn

from mmdet.core import auto_fp16
from ..registry import NECKS

from .sfpn import SFPN

@NECKS.register_module
class TSFPN(nn.Module):
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
        super(TSFPN, self).__init__()
        args_dict = locals()
        args_dict.pop('self')
        args_dict.pop('__class__')
        self.branch_cls = SFPN(**args_dict)
        self.branch_reg = SFPN(**args_dict)
        self.branch_anchor = SFPN(**args_dict)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        self.branch_cls.init_weights()
        self.branch_reg.init_weights()
        self.branch_anchor.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        branch_cls_outs = self.branch_cls(inputs)
        branch_reg_outs = self.branch_reg(inputs)
        branch_anchor_outs = self.branch_anchor(inputs)
        outs = [i for i in zip(branch_cls_outs, branch_reg_outs, branch_anchor_outs)]
        return tuple(outs)
