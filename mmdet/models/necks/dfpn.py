import torch.nn as nn
import torch.nn.functional as F
from mmdet.ops import ConvModule

from mmdet.core import auto_fp16
from ..builder import NECKS
from .fpn import FPN

import matplotlib.pyplot as plt
import torch
import os

@NECKS.register_module()
class DFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 extra_convs_on_inputs_reg=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 save_outs=False,
                 freeze=False):
        super(DFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg, save_outs=save_outs,freeze=freeze)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.dfpn_convs = nn.ModuleList()
        self.dfpn_convs.append(ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False))
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            dfpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.extra_convs_on_inputs_reg = extra_convs_on_inputs_reg
            self.downsample_convs.append(d_conv)
            self.dfpn_convs.append(dfpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and extra_convs_on_inputs_reg:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_dfpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.dfpn_convs.append(extra_dfpn_conv)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.save_outs:
            laterals_ori = [l.clone() for l in laterals]
        laterals_down_top = [l.clone() for l in laterals]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        cls_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            laterals_down_top[i + 1] += self.downsample_convs[i](laterals_down_top[i])

        # reg_outs = []
        # reg_outs.append(laterals_down_top[0])
        reg_outs = [self.dfpn_convs[i](laterals_down_top[i]) for i in range(used_backbone_levels)]
        # part 3: add extra levels
        # top-down
        if self.num_outs > len(cls_outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    cls_outs.append(F.max_pool2d(cls_outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    cls_outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    cls_outs.append(self.fpn_convs[used_backbone_levels](cls_outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        cls_outs.append(self.fpn_convs[i](F.relu(cls_outs[-1])))
                    else:
                        cls_outs.append(self.fpn_convs[i](cls_outs[-1]))
        # down-top
        if self.num_outs > len(reg_outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    reg_outs.append(F.max_pool2d(reg_outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs_reg:
                    orig = inputs[self.backbone_end_level - 1]
                    reg_outs.append(self.dfpn_convs[used_backbone_levels](orig))
                else:
                    reg_outs.append(self.dfpn_convs[used_backbone_levels](reg_outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        reg_outs.append(self.dfpn_convs[i](F.relu(reg_outs[-1])))
                    else:
                        reg_outs.append(self.dfpn_convs[i](reg_outs[-1]))
        if self.save_outs:
            res = dict()
            res['laterals'] = laterals
            res['laterals_ori'] = laterals_ori
            res['cls_outs'] = cls_outs
            res['reg_outs'] = reg_outs
            save_idx = 1
            save_path_base = 'mytest/dfpn_outs.pth'
            save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
            while os.path.exists(save_path):
                save_idx += 1
                save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
            torch.save(res, save_path)

        return tuple([(cls_out, reg_out) for cls_out, reg_out in zip(cls_outs, reg_outs)])
