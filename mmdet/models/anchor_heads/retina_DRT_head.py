import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.ops import ConvModule, MaskedConv2d, DeformConv
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead

import torch
import os
from mmdet.core import (delta2bbox, multi_apply, multiclass_nms)
class FeatureAdaptionCls(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_offset_in_channels=2,
                 conv_offset_kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaptionCls, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            conv_offset_in_channels, deformable_groups * offset_channels, conv_offset_kernel_size, padding=conv_offset_kernel_size//2,  bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape, save_out=False):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        if save_out:
            return x, offset
        else:
            return x

@HEADS.register_module
class RetinaDRTHead(GuidedAnchorHead):
    """Guided-Anchor-based RetinaNet head."""

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 freeze=False,
                 save_outs=False,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.save_outs = save_outs
        # print('conv_cfg: ', conv_cfg)
        super(RetinaDRTHead, self).__init__(num_classes, in_channels, **kwargs)

        if freeze:
            for c in [self.cls_convs, self.reg_convs, self.conv_loc, self.conv_shape,
                      self.feature_adaption_cls, self.feature_adaption_reg]:
                for p in c.parameters():
                    p.requires_grad = False

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)

        self.cls_feat_enhance = nn.Conv2d(self.feat_channels, self.deformable_groups, 3, stride=1, padding=1)

        self.feature_adaption_cls = FeatureAdaptionCls(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            conv_offset_in_channels=self.deformable_groups,
            conv_offset_kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.feature_adaption_reg = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.retina_cls = MaskedConv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = MaskedConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)


    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        normal_init(self.cls_feat_enhance, std=0.01)

        self.feature_adaption_cls.init_weights()
        self.feature_adaption_reg.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        cls_feat_enhance_pre = self.cls_feat_enhance(cls_feat)

        if self.save_outs:
            cls_feat, offset_cls = self.feature_adaption_cls(cls_feat, cls_feat_enhance_pre, self.save_outs)
            reg_feat, offset_reg = self.feature_adaption_reg(reg_feat, shape_pred, self.save_outs)
        else:
            cls_feat = self.feature_adaption_cls(cls_feat, cls_feat_enhance_pre, self.save_outs)
            reg_feat = self.feature_adaption_reg(reg_feat, shape_pred, self.save_outs)

        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.retina_cls(cls_feat, mask)
        bbox_pred = self.retina_reg(reg_feat, mask)

        if self.save_outs:
            return cls_score, bbox_pred, shape_pred, loc_pred, offset_cls, offset_reg
        else:
            return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, feats):
        if self.training:
            return multi_apply(self.forward_single, feats)
        else:
            if self.save_outs:
                cls_scores, bbox_preds, shape_preds_reg, loc_preds, offsets_cls, offsets_reg = multi_apply(self.forward_single, feats)
                res = dict()
                res['offsets_cls'] = offsets_cls
                res['offsets_reg'] = offsets_reg
                save_idx = 1
                save_path_base = 'mytest/ga_retina_feature.pth'
                save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
                while os.path.exists(save_path):
                    save_idx += 1
                    save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
                torch.save(res, save_path)
            else:
                cls_scores, bbox_preds, shape_preds_reg, loc_preds = multi_apply(self.forward_single, feats)
            return cls_scores, bbox_preds, shape_preds_reg, loc_preds

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          mlvl_masks,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       mlvl_anchors,
                                                       mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and bbox_pred
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
                # scores = cls_score
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        if len(mlvl_bboxes) == 0:
            mlvl_bboxes.append(torch.tensor([[0., 0., 1e-10, 1e-10]]))
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        if len(mlvl_scores) == 0:
            mlvl_scores.append(torch.tensor([[0.]]))
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels