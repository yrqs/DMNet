import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn import normal_init

from mmdet.core import auto_fp16
from mmdet.ops import ConvModule
from ..registry import NECKS

import torch
import os
import random

@NECKS.register_module
class DFPN3(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = DFPN3(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print('outputs[{}].shape = {!r}'.format(i, outputs[i].shape))
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

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
                 r=16,
                 save_outs=False,
                 freeze=False):
        super(DFPN3, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.save_outs = save_outs
        self.r = r
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.channel_attention_cls = torch.nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels//self.r),
            nn.ReLU(True),
            nn.Linear(self.out_channels//self.r, self.out_channels),
            nn.Sigmoid()
        )
        self.channel_attention_reg = torch.nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels//self.r),
            nn.ReLU(True),
            nn.Linear(self.out_channels//self.r, self.out_channels),
            nn.Sigmoid()
        )

        self.spatial_attention_cls = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.spatial_attention_reg = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        if freeze:
            for idx, p in enumerate(self.parameters()):
                p.requires_grad = False
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for n in [self.channel_attention_cls, self.channel_attention_reg, self.spatial_attention_cls, self.spatial_attention_reg]:
            for m in n:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
                if isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if self.save_outs:
            laterals_ori = [l.clone() for l in laterals]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        outs_cls = []
        outs_reg = []
        if self.save_outs:
            spatial_attentions_cls = []
            spatial_attentions_reg = []
        for out in outs:
            out_avg = self.avg_pool(out).squeeze(-1).squeeze(-1)
            channel_attention_cls = self.channel_attention_cls(out_avg).unsqueeze(-1).unsqueeze(-1)
            channel_attention_reg = self.channel_attention_reg(out_avg).unsqueeze(-1).unsqueeze(-1)
            out_cls = out * channel_attention_cls
            out_reg = out * channel_attention_reg
            out_channel_avg_cls = torch.mean(out_cls, dim=1, keepdim=True)
            out_channel_avg_reg = torch.mean(out_reg, dim=1, keepdim=True)
            spatial_attention_cls = self.spatial_attention_cls(out_channel_avg_cls)
            spatial_attention_reg = self.spatial_attention_reg(out_channel_avg_reg)
            if self.save_outs:
                spatial_attentions_cls.append(spatial_attention_cls)
                spatial_attentions_reg.append(spatial_attention_reg)
            out_cls = out_cls * spatial_attention_cls
            out_reg = out_reg * spatial_attention_reg
            outs_cls.append(out_cls)
            outs_reg.append(out_reg)

        if self.save_outs:
            res = dict()
            res['laterals'] = laterals
            res['laterals_ori'] = laterals_ori
            res['fpn_outs'] = outs
            res['spatial_attentions_cls'] = spatial_attentions_cls
            res['spatial_attentions_reg'] = spatial_attentions_reg
            res['outs_cls'] = outs_cls
            res['outs_reg'] = outs_reg
            save_idx = 1
            save_path_base = 'mytest/dfpn3_outs.pth'
            save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
            while os.path.exists(save_path):
                save_idx += 1
                save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
            torch.save(res, save_path)

        return tuple([(cls_out, reg_out) for cls_out, reg_out in zip(outs_cls, outs_reg)])