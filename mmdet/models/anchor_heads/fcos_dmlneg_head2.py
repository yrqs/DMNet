import torch
import torch.nn as nn
from mmcv.cnn import normal_init, constant_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from mmdet.ops import ConvModule, Scale
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob

import torch.nn.functional as F

INF = 1e8

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class MPC(nn.Module):
    def __init__(self,
                 num_layer,
                 in_channels,
                 out_channels):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(nn.Linear(in_channels, out_channels))
            if i < num_layer-1:
                self.layers.append(nn.ReLU())

        for m in self.layers:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class DMLNegHead(nn.Module):
    def __init__(self,
                 emb_module,
                 output_channels,
                 emb_channels,
                 num_modes,
                 sigma,
                 cls_norm,
                 neg_scope=2.0,
                 beta=0.3,
                 neg_num_modes=3,
                 neg_rep_fc_layers=3,
                 freeze=False):
        assert num_modes==1
        super().__init__()
        self.output_channels = output_channels
        self.num_modes = num_modes
        self.sigma = sigma
        self.emb_module = emb_module
        self.emb_channels = emb_channels
        self.rep_fc = nn.Linear(1, output_channels * num_modes * emb_channels[-1])
        self.cls_norm = cls_norm
        self.beta = beta
        self.neg_num_modes = neg_num_modes
        self.neg_scope = neg_scope
        self.representations = nn.Parameter(
            torch.FloatTensor(self.output_channels, self.num_modes, self.emb_channels[-1]),
            requires_grad=False)
        self.neg_rep_gen = nn.ModuleList()
        for i in range(neg_num_modes):
            self.neg_rep_gen.append(MPC(neg_rep_fc_layers, emb_channels[-1], emb_channels[-1]))
        # self.neg_offset_fc = nn.Linear(self.emb_channels[-1], self.emb_channels[-1] * neg_num_modes)

        normal_init(self.rep_fc, std=0.01)
        # normal_init(self.neg_offset_fc, std=0.01)
        # constant_init(self.neg_offset_fc, 0)

        if freeze:
            for c in [self.neg_offset_fc]:
                for p in c.parameters():
                    p.requires_grad = False

    def forward(self, x, save_outs=False):
        emb_vectors = self.emb_module(x)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        if self.training:
            reps = self.rep_fc(torch.tensor(1.0).to(x.device).unsqueeze(0)).squeeze(0)
            reps  = reps.view(self.output_channels, self.num_modes, self.emb_channels[-1])
            reps = F.normalize(reps, p=2, dim=2)
            self.representations.data = reps.detach()
        else:
            reps = self.representations.detach()

        # neg_offset = self.neg_offset_fc(reps.squeeze(1)).view(reps.size(0), self.neg_num_modes, reps.size(-1))
        # reps_neg = neg_offset + reps.expand_as(neg_offset)
        reps_neg_list = []
        for nrg in self.neg_rep_gen:
            reps_neg_list.append(nrg(reps.squeeze(1)).unsqueeze(1))
        reps_neg = torch.cat(reps_neg_list, dim=1)
        reps_neg = F.normalize(reps_neg, p=2, dim=2)

        distances = emb_vectors.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        distances = distances.expand(-1, -1, -1, self.output_channels, self.num_modes, -1)
        distances = torch.sqrt(((distances - reps)**2).sum(-1)).permute(0, 3, 4, 1, 2).contiguous()

        distances_neg = emb_vectors.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        distances_neg = distances_neg.expand(-1, -1, -1, self.output_channels, self.neg_num_modes, -1)
        distances_neg = torch.sqrt(((distances_neg - reps_neg)**2).sum(-1)).permute(0, 3, 4, 1, 2).contiguous()

        probs_neg = torch.exp(-distances_neg**2/(2.0*self.sigma**2))
        cls_score_neg = probs_neg.max(dim=2)[0]

        probs_ori = torch.exp(-(distances)**2/(2.0*self.sigma**2))
        probs_ori = probs_ori.max(dim=2)[0]
        probs = torch.exp(-(distances+self.beta*(F.relu(self.neg_scope-distances_neg.min(dim=2, keepdim=True)[0])))**2/(2.0*self.sigma**2))
        if self.cls_norm:
            probs_sumj = probs.sum(2)
            probs_sumij = probs_sumj.sum(1, keepdim=True)
            cls_score = probs_sumj / probs_sumij
        else:
            cls_score = probs.max(dim=2)[0]

        # probs_bg = torch.sub(1, probs.max(1)[0].max(1, keepdim=True)[0])
        # if self.training:
        #     probs_bg = torch.sub(1, probs_cls.max(1)[0].max(1, keepdim=True)[0])
        # else:
        #     probs_bg = torch.zeros(probs_fg.shape[0], 1, probs_fg.shape[2], probs_fg.shape[3]).to(probs_fg.device)

        # cls_score = torch.cat((probs_bg, probs_fg), 1)

        if save_outs:
            return cls_score, cls_score_neg, distances, distances_neg, probs_ori, emb_vectors, reps, reps_neg
        return cls_score, cls_score_neg, distances, distances_neg, probs_ori


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
class FCOSDMLNegHead2(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 neg_sample_thresh=0.2,
                 cls_emb_head_cfg=dict(
                     emb_channels=(256, 128),
                     num_modes=1,
                     sigma=0.5,
                     cls_norm=False,
                     beta=0.3,
                     neg_num_modes=3),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     neg_rep_fc_layers=3,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0),
                 loss_emb_neg=dict(type='RepMetLoss', alpha=0.10, loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super().__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius

        self.neg_sample_thresh = neg_sample_thresh
        self.cls_emb_head_cfg = cls_emb_head_cfg
        self.loss_emb = build_loss(loss_emb)
        self.loss_emb_neg = build_loss(loss_emb_neg)

        self._init_layers()

    def _init_layers(self):
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
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        # self.fcos_cls = nn.Conv2d(
        #     self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        emb_module = build_emb_module(self.feat_channels, self.cls_emb_head_cfg['emb_channels'],
                                      kernel_size=3, padding=1, stride=1)
        self.fcos_cls = DMLNegHead(emb_module, self.num_classes-1, **self.cls_emb_head_cfg)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        # bias_cls = bias_init_with_prob(0.01)
        # normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score, cls_score_neg, distance, distance_neg, probs_ori = self.fcos_cls(cls_feat)
        # cls_score = self.fcos_cls(cls_feat)
        cls_score = inverse_sigmoid(cls_score)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        if self.training:
            return cls_score, bbox_pred, centerness, distance, cls_score_neg, distance_neg, probs_ori
        else:
            return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses', 'distances', 'cls_scores_neg', 'distances_neg', 'probs_ori'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             distances,
             cls_scores_neg,
             distances_neg,
             probs_ori,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        flatten_distances = [dis.permute(0, 3, 4, 1, 2).flatten(0, 2) for dis in distances]
        flatten_distances = torch.cat(flatten_distances, dim=0)
        if num_pos > 0:
            flatten_distances_pos = flatten_distances[flatten_labels > 0]
            flatten_labels_pos = flatten_labels[flatten_labels > 0].view(-1, 1)
            loss_emb = self.loss_emb(
                flatten_distances_pos,
                flatten_labels_pos,
                avg_factor=num_pos + num_imgs
            )
        else:
            loss_emb = torch.tensor(0.0).to(loss_bbox.device)

        flatten_distances_neg = [dis.permute(0, 3, 4, 1, 2).flatten(0, 2) for dis in distances_neg]
        flatten_distances_neg = torch.cat(flatten_distances_neg, dim=0)
        flatten_probs_ori = [
            prob.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for prob in probs_ori
        ]
        flatten_probs_ori = torch.cat(flatten_probs_ori)

        labels_one_hot = flatten_labels.new_full((flatten_labels.shape[0], self.cls_out_channels + 1), 0,
                                                 dtype=torch.long)
        labels_one_hot = labels_one_hot.scatter(1, flatten_labels.unsqueeze(1), 1)[:, 1:]

        neg_labels, num_samples_neg, neg_labels_one_hot = self.get_neg_labels(flatten_probs_ori, flatten_labels)

        flatten_distance_cat = torch.cat([flatten_distances, flatten_distances_neg.min(dim=-1, keepdim=True)[0]], dim=-1)

        distance_cat_pos = flatten_distance_cat[labels_one_hot==1].unsqueeze(-1)
        labels_pos = distance_cat_pos.new_full((distance_cat_pos.shape[0], 1), 1, dtype=torch.long)
        loss_emb_pos = self.loss_emb_neg(
            distance_cat_pos,
            labels_pos,
            avg_factor=num_pos + num_imgs
        )

        distance_cat_neg = flatten_distance_cat[neg_labels_one_hot==1].unsqueeze(-1)
        labels_neg = distance_cat_neg.new_full((distance_cat_neg.shape[0], 1), 2, dtype=torch.long)
        loss_emb_neg = self.loss_emb_neg(
            distance_cat_neg,
            labels_neg,
            avg_factor=num_samples_neg + num_imgs
        )

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_emb=loss_emb,
            loss_emb_pos=loss_emb_pos,
            loss_emb_neg=loss_emb_neg,
        )

    def get_neg_labels(self, cls_score, labels):
        pred_cls_score, pred_cls_id = cls_score.max(dim=-1)
        neg_labels = pred_cls_id.clone() + 1
        neg_labels[neg_labels==labels] = 0
        neg_labels[pred_cls_score<self.neg_sample_thresh] = 0
        num_samples = int((neg_labels > 0).sum())
        neg_labels = neg_labels.reshape(-1, 1)
        neg_labels_one_hot = neg_labels.new_full((neg_labels.shape[0], cls_score.shape[-1]+1), 0, dtype=torch.long)
        neg_labels_one_hot = neg_labels_one_hot.scatter(1, neg_labels, 1)[:,  1:]
        return neg_labels, num_samples, neg_labels_one_hot

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)