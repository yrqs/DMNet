import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init, constant_init

from mmdet.ops import ConvModule
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .anchor_head import AnchorHead
import torch
from ..builder import build_loss
import torch.nn.functional as F

from mmdet.core import (anchor_target, force_fp32, multi_apply)


@HEADS.register_module
class RetinaDMLHead2(AnchorHead):
    """
    An anchor-based head used in [1]_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> import torch
        >>> self = RetinaDMLHead2(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes - 1)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 emb_sizes=(2048, 1024),
                 num_modes=5,
                 sigma=0.5,
                 loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.emb_sizes = emb_sizes
        self.num_modes = num_modes
        self.sigma = sigma
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(RetinaDMLHead2, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
        self.loss_emb = build_loss(loss_emb)

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
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        self.rep = nn.Linear(1, (self.num_classes-1) * self.num_modes * self.emb_sizes[-1])
        self.representations = nn.Parameter(
            torch.FloatTensor(self.num_classes-1, self.num_modes, self.emb_sizes[-1]),
            requires_grad=False
        )
        self.emb = nn.ModuleList()
        for i in range(len(self.emb_sizes)):
            if i == 0:
                self.emb.append(nn.Conv2d(self.feat_channels, self.emb_sizes[i], 3, padding=1, stride=1)),
                self.emb.append(nn.BatchNorm2d(self.emb_sizes[i])),
                # self.emb.append(self.relu)
            else:
                self.emb.append(nn.Conv2d(self.emb_sizes[i-1], self.emb_sizes[i], 3, padding=1, stride=1)),
                if i != len(self.emb_sizes) - 1:
                    self.emb.append(nn.BatchNorm2d(self.emb_sizes[i])),
                    # self.emb.append(self.relu)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.retina_reg, std=0.01)

        for m in self.emb:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        normal_init(self.rep, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        if self.training:
            reps = self.rep(torch.tensor(1.0).to(reg_feat.device).unsqueeze(0)).squeeze(0)
            reps  = reps.view((self.num_classes-1), self.num_modes, self.emb_sizes[-1])
            reps = F.normalize(reps, p=2, dim=2)
            self.representations.data = reps.detach()
        else:
            reps = self.representations.detach()

        emb_vectors = cls_feat

        for e in self.emb:
            emb_vectors = e(emb_vectors)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        distances = emb_vectors.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        distances = distances.expand(-1, -1, -1, self.num_classes-1, self.num_modes, -1)

        distances = torch.sqrt(((distances - reps)**2).sum(-1)).permute(0, 3, 4, 1, 2).contiguous()

        probs_cls = torch.exp(-distances**2/(2.0*self.sigma**2))
        # print(probs_cls.size())
        probs_cls_sumj = probs_cls.sum(2)
        probs_cls_sumij = probs_cls_sumj.sum(1, keepdim=True)
        probs_fg = probs_cls_sumj / probs_cls_sumij

        probs_bg = torch.sub(1, probs_cls.max(1)[0].max(1, keepdim=True)[0])

        cls_score = torch.cat((probs_bg, probs_fg), 1)

        bbox_pred = self.retina_reg(reg_feat)

        if self.training:
            return cls_score, bbox_pred, distances
        else:
            return cls_score.log(), bbox_pred

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        # labels = labels.reshape(-1)
        # label_weights = label_weights.reshape(-1)
        # cls_score = cls_score.permute(0, 2, 3,
        #                               1).reshape(-1, self.cls_out_channels)
        # loss_cls = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        loss_cls_all = F.nll_loss(cls_score.log(), labels, None, None, -100, None, 'none') * label_weights
        # print(loss_cls_all.size())
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'distances'))
    def loss(self,
             cls_scores,
             bbox_preds,
             distances,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)

        losses_emb = []
        num_total_samples_emb = sum([int((_>0).sum()) for _ in labels_list])
        for i in range(len(distances)):
            loss_emb = self.loss_emb_single(
                distances[i],
                labels_list[i],
                label_weights_list[i],
                num_total_samples=num_total_samples_emb)
            losses_emb.append(loss_emb)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_emb=losses_emb)

    def loss_emb_single(self, distance, label, label_weights, num_total_samples):
        # filter out negative samples
        # print(label.size())
        if label.dim() == 1:
            label = label.unsqueeze(0)
            label_weights = label_weights.unsqueeze(0)
        distance = distance.view(distance.size(0), distance.size(1), distance.size(2), -1).permute(0, 3, 1, 2)
        # label_ = label.view(distance.size(0), 1, -1)
        pos_inds = label > 0

        distance_pos = distance[pos_inds]
        # print(distance_pos.size())
        label_weights_pos = label_weights[label>0]
        label_pos = label[pos_inds].view(-1, 1)
        # label_weights_pos = label_weights[label>0]
        # label_pos = label[label>0].view(distance.size(0), 1, -1)
        loss_emb = self.loss_emb(
            distance_pos,
            label_pos,
            label_weights_pos,
            # avg_factor=num_total_samples*10)
            # avg_factor=max(1, int(distance_pos.size(0))))
            avg_factor=max(1, num_total_samples))
        # avg_factor=max(1, int(distance.size(1))))
        return loss_emb