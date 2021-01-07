import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.ops import ConvModule
from ..registry import NECKS

import torch
import os

@NECKS.register_module
class DFPN2(nn.Module):
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
        >>> self = DFPN2(in_channels, 11, len(in_channels)).eval()
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
                 save_outs=False,
                 freeze=False,
                 freeze_ratio=10):
        super(DFPN2, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.save_outs = save_outs
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

        self.lateral_convs_cls = nn.ModuleList()
        self.fpn_convs_cls = nn.ModuleList()
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
            self.lateral_convs_cls.append(l_conv)
            self.fpn_convs_cls.append(fpn_conv)

        self.lateral_convs_reg = nn.ModuleList()
        self.fpn_convs_reg = nn.ModuleList()
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
            self.lateral_convs_reg.append(l_conv)
            self.fpn_convs_reg.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv_cls = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                extra_fpn_conv_reg = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs_cls.append(extra_fpn_conv_cls)
                self.fpn_convs_reg.append(extra_fpn_conv_reg)

        if freeze:
            for idx, p in enumerate(self.parameters()):
                if idx % (int(freeze_ratio*10)) == 0:
                    p.requires_grad = False
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals_cls = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs_cls)
        ]
        laterals_reg = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs_reg)
        ]

        if self.save_outs:
            laterals_cls_ori = [l.clone() for l in laterals_cls]
            laterals_reg_ori = [l.clone() for l in laterals_reg]

        # build top-down path
        used_backbone_levels = len(laterals_cls)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals_cls[i - 1].shape[2:]
            laterals_cls[i - 1] += F.interpolate(
                laterals_cls[i], size=prev_shape, mode='nearest')
            laterals_reg[i - 1] += F.interpolate(
                laterals_reg[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs_cls = [
            self.fpn_convs_cls[i](laterals_cls[i]) for i in range(used_backbone_levels)
        ]
        outs_reg = [
            self.fpn_convs_reg[i](laterals_reg[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs_cls):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs_cls.append(F.max_pool2d(outs_cls[-1], 1, stride=2))
                    outs_reg.append(F.max_pool2d(outs_reg[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs_cls.append(self.fpn_convs_cls[used_backbone_levels](orig))
                    outs_reg.append(self.fpn_convs_reg[used_backbone_levels](orig))
                else:
                    outs_cls.append(self.fpn_convs_cls[used_backbone_levels](outs_cls[-1]))
                    outs_reg.append(self.fpn_convs_reg[used_backbone_levels](outs_reg[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs_cls.append(self.fpn_convs_cls[i](F.relu(outs_cls[-1])))
                        outs_reg.append(self.fpn_convs_reg[i](F.relu(outs_reg[-1])))
                    else:
                        outs_cls.append(self.fpn_convs_cls[i](outs_cls[-1]))
                        outs_reg.append(self.fpn_convs_reg[i](outs_reg[-1]))
        if self.save_outs:
            res = dict()
            res['laterals_cls'] = laterals_cls
            res['laterals_reg'] = laterals_reg
            res['laterals_ori_cls'] = laterals_cls_ori
            res['laterals_ori_reg'] = laterals_reg_ori
            res['outs_cls'] = outs_cls
            res['outs_reg'] = outs_reg
            save_idx = 1
            save_path_base = 'mytest/dfpn2_outs.pth'
            save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
            while os.path.exists(save_path):
                save_idx += 1
                save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
            torch.save(res, save_path)

        return tuple([(cls_out, reg_out) for cls_out, reg_out in zip(outs_cls, outs_reg)])
