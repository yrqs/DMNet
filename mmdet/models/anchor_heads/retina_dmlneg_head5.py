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

from mmdet.core import (anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from mmdet.models.subnets.MLP import MLP

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class DMLHead(nn.Module):
    def __init__(self,
                 emb_module,
                 output_channels,
                 emb_channels,
                 num_modes,
                 sigma,
                 cls_norm,
                 beta=0.3,
                 neg_num_modes=3,
                 freeze=False):
        assert num_modes==1
        super().__init__()
        self.output_channels = output_channels
        self.num_modes = num_modes
        self.sigma = sigma
        self.emb_module = emb_module
        self.emb_channels = emb_channels
        # self.rep_fc = nn.Linear(1, output_channels * num_modes * emb_channels[-1])
        self.cls_norm = cls_norm
        # self.representations = nn.Parameter(
        #     torch.FloatTensor(self.output_channels, self.num_modes, self.emb_channels[-1]),
        #     requires_grad=False)

        self.beta = beta
        self.neg_num_modes = neg_num_modes
        reps = nn.Embedding(self.output_channels*self.num_modes, self.emb_channels[-1])
        normal_init(reps, std=0.01)
        self.representations = reps.weight

        self.neg_offsets_generators = nn.ModuleList()

        self.neg_offset_fc = nn.Linear(self.emb_channels[-1], self.emb_channels[-1] * neg_num_modes)
        normal_init(self.neg_offset_fc, std=0.01)
        # normal_init(self.rep_fc, std=0.01)
        # constant_init(self.neg_offset_fc, 0)

        if freeze:
            for c in [self.neg_offset_fc]:
                for p in c.parameters():
                    p.requires_grad = False

    def forward(self, x, save_outs=False):
        emb_vectors = self.emb_module(x)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        # if self.training:
        #     reps = self.rep_fc(torch.tensor(1.0).to(x.device).unsqueeze(0)).squeeze(0)
        #     reps  = reps.view(self.output_channels, self.num_modes, self.emb_channels[-1])
        #     reps = F.normalize(reps, p=2, dim=2)
        #     self.representations.data = reps.detach()
        # else:
        #     reps = self.representations.detach()

        reps = self.representations.view(self.output_channels, self.num_modes, self.emb_channels[-1])
        reps = F.normalize(reps, p=2, dim=2)

        emb_vectors_ex = emb_vectors.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        emb_vectors_ex = emb_vectors_ex.expand(-1, -1, -1, self.output_channels, self.num_modes, -1)

        distances = torch.sqrt(((emb_vectors_ex - reps)**2).sum(-1)).permute(0, 3, 4, 1, 2).contiguous()
        probs_ori = torch.exp(-(distances)**2/(2.0*self.sigma**2))
        probs_ori = probs_ori.max(dim=2)[0]

        neg_offset = self.neg_offset_fc(reps.squeeze(1)).view(reps.size(0), self.neg_num_modes, reps.size(-1))
        neg_reps = neg_offset + reps.expand_as(neg_offset)
        neg_reps = F.normalize(neg_reps, p=2, dim=2)

        distances_neg = torch.sqrt(((emb_vectors_ex - neg_reps)**2).sum(-1)).permute(0, 3, 4, 1, 2).contiguous()

        probs = torch.exp(-(distances + self.beta*(2-distances_neg.min(dim=2, keepdim=True)[0]))**2/(2.0*self.sigma**2))
        if self.cls_norm:
            probs_sumj = probs.sum(2)
            probs_sumij = probs_sumj.sum(1, keepdim=True)
            cls_score = probs_sumj / probs_sumij
        else:
            cls_score = probs.max(dim=2)[0]

        cls_score = torch.sqrt(probs_ori) * cls_score

        if save_outs:
            return cls_score, distances, probs_ori, emb_vectors, reps
        return cls_score, distances, distances_neg, probs_ori


def build_emb_module(input_channels, emb_channels, kernel_size=1, padding=0, stride=0):
    emb_list = []
    for i in range(len(emb_channels)):
        if i == 0:
            emb_list.append(nn.Conv2d(input_channels, emb_channels[i], kernel_size, padding=padding, stride=stride)),
            emb_list.append(nn.BatchNorm2d(emb_channels[i])),
            # self.emb.append(self.relu)
        else:
            emb_list.append(nn.Conv2d(emb_channels[i - 1], emb_channels[i], kernel_size, padding=padding, stride=stride)),
            if i != len(emb_channels) - 1:
                emb_list.append(nn.BatchNorm2d(emb_channels[i])),
                # self.emb.append(self.relu)

    for m in emb_list:
        if isinstance(m, nn.Conv2d):
            normal_init(m, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            constant_init(m, 1)

    return nn.Sequential(*tuple(emb_list))


@HEADS.register_module
class RetinaDMLNegHead5(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 neg_sample_thresh=0.2,
                 cls_emb_head_cfg=dict(
                     emb_channels=(256, 128),
                     num_modes=1,
                     sigma=0.5,
                     cls_norm=False,
                     beta=0.3,
                     neg_num_modes=3,
                 ),
                 loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0),
                 loss_emb_neg=dict(type='RepMetLoss', alpha=0.10, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.neg_sample_thresh = neg_sample_thresh
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        self.cls_emb_head_cfg = cls_emb_head_cfg

        super().__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, **kwargs)
        self.loss_emb = build_loss(loss_emb)
        self.loss_emb_neg = build_loss(loss_emb_neg)

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

        emb_module = build_emb_module(self.feat_channels, self.cls_emb_head_cfg['emb_channels'],
                                      kernel_size=3, padding=1, stride=1)
        self.cls_head = DMLHead(emb_module, self.num_classes-1, **self.cls_emb_head_cfg)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score, distance, distance_neg, probs_ori = self.cls_head(cls_feat)
        cls_score = inverse_sigmoid(cls_score)

        bbox_pred = self.retina_reg(reg_feat)

        if self.training:
            return cls_score, bbox_pred, distance, distance_neg, probs_ori
        else:
            return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'distances', 'distances_neg', 'probs_ori_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             distances,
             distances_neg,
             probs_ori_preds,
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

        num_total_samples_neg = 0
        neg_labels_list = []
        neg_labels_one_hot_list = []
        for prob_ori, labels in zip(probs_ori_preds, labels_list):
            neg_labels, num_samples_neg, neg_labels_one_hot = self.get_neg_labels_single(prob_ori, labels)
            # neg_labels, num_samples_neg, neg_labels_one_hot = self.get_neg_labels_single(cls_score, labels)
            neg_labels_list.append(neg_labels)
            neg_labels_one_hot_list.append(neg_labels_one_hot)
            num_total_samples_neg += num_samples_neg
            # print('num_samples_neg: ', num_samples_neg)
        num_total_samples_neg = max(1, num_total_samples_neg)

        labels_one_hot_list = []
        for i, labels in enumerate(labels_list):
            labels_flatten = labels.reshape(-1, 1)
            labels_one_hot = labels_flatten.new_full((labels_flatten.shape[0], self.cls_out_channels + 1), 0,
                                                     dtype=torch.long)
            labels_one_hot = labels_one_hot.scatter(1, labels_flatten, 1)[:, 1:]
            labels_one_hot_list.append(labels_one_hot)

        losses_emb_neg = []
        losses_emb_pos = []
        losses_neg_cluster = []
        num_total_samples_emb_neg = sum([int((_ > 0).sum()) for _ in neg_labels_list])
        for i in range(len(cls_scores)):
            loss_emb_pos, loss_emb_neg, loss_neg_cluster = self.loss_emb_neg_single(
                distances[i],
                distances_neg[i],
                labels_list[i],
                neg_labels_list[i],
                label_weights_list[i],
                labels_one_hot_list[i],
                neg_labels_one_hot_list[i],
                num_total_samples_pos=num_total_samples_emb,
                num_total_samples_neg=num_total_samples_emb_neg)
            losses_emb_neg.append(loss_emb_neg)
            losses_emb_pos.append(loss_emb_pos)
            losses_neg_cluster.append(loss_neg_cluster)

        return dict(loss_cls=losses_cls,
                    loss_bbox=losses_bbox,
                    loss_emb=losses_emb,
                    loss_emb_neg=losses_emb_neg,
                    loss_emb_pos=losses_emb_pos,
                    losses_neg_cluster=losses_neg_cluster,
                    )

    def loss_emb_single(self, distance, label, label_weights, num_total_samples):
        # filter out negative samples
        if label.dim() == 1:
            label = label.unsqueeze(0)
            label_weights = label_weights.unsqueeze(0)
        distance = distance.view(distance.size(0), distance.size(1), distance.size(2), -1).permute(0, 3, 1, 2)
        pos_inds = label > 0

        distance_pos = distance[pos_inds]
        label_weights_pos = label_weights[label>0]
        label_pos = label[pos_inds].view(-1, 1)
        loss_emb = self.loss_emb(
            distance_pos,
            label_pos,
            label_weights_pos,
            avg_factor=max(1, num_total_samples))
        return loss_emb

    def loss_emb_neg_single(self, distance, distance_neg, labels, labels_neg, label_weights, labels_ont_hot,
                            neg_labels_one_hot, num_total_samples_pos, num_total_samples_neg):
        distance = distance.flatten(3).permute(0, 3, 1, 2).flatten(0, 1)
        distance_neg = distance_neg.flatten(3).permute(0, 3, 1, 2).min(dim=-1, keepdim=True)[0].flatten(0, 1)

        distance_cat = torch.cat([distance, distance_neg], dim=-1)

        distance_cat_pos = distance_cat[labels_ont_hot==1].unsqueeze(-1)
        labels = labels.reshape(-1)
        pos_inds = labels > 0
        label_weights_pos = label_weights.reshape(-1)[pos_inds]
        labels_pos = distance_cat_pos.new_full((distance_cat_pos.shape[0], 1), 1, dtype=torch.long)
        loss_emb_pos = self.loss_emb_neg(
            distance_cat_pos,
            labels_pos,
            label_weights_pos,
            avg_factor=max(1, num_total_samples_pos))

        distance_cat_neg = distance_cat[neg_labels_one_hot==1].unsqueeze(-1)
        labels_neg = labels_neg.reshape(-1)
        neg_inds = labels_neg > 0
        label_weights_neg = label_weights.reshape(-1)[neg_inds]
        labels_neg = distance_cat_neg.new_full((distance_cat_neg.shape[0], 1), 2, dtype=torch.long)
        loss_emb_neg = self.loss_emb_neg(
            distance_cat_neg,
            labels_neg,
            label_weights_neg,
            avg_factor=max(1, num_total_samples_neg))

        loss_neg_cluster = distance_neg[neg_labels_one_hot == 1].sum() / max(1, num_total_samples_neg)

        return loss_emb_pos, loss_emb_neg, loss_neg_cluster

    def get_neg_labels_single(self, cls_score, labels):
        cls_score = cls_score.flatten(2).permute(0, 2, 1)
        pred_cls_score, pred_cls_id = cls_score.max(dim=-1)
        neg_labels = pred_cls_id.clone() + 1
        neg_labels[neg_labels==labels] = 0
        neg_labels[pred_cls_score<self.neg_sample_thresh] = 0
        num_samples = int((neg_labels > 0).sum())
        neg_labels = neg_labels.reshape(-1, 1)
        neg_labels_one_hot = neg_labels.new_full((neg_labels.shape[0], cls_score.shape[-1]+1), 0, dtype=torch.long)
        neg_labels_one_hot = neg_labels_one_hot.scatter(1, neg_labels, 1)[:,  1:]
        return neg_labels, num_samples, neg_labels_one_hot

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
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
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels