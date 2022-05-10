import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from mmdet.ops.scale_grad import scale_tensor_gard
from mmdet.models.subnets.channel_transform import ChannelAttention
from mmdet.utils.show_feature import show_feature

@HEADS.register_module
class FSBBoxHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 dropout_ratio=None,
                 add_center_loss=False,
                 center_loss_weight=0.1,
                 cls_w_l1_regular=False,
                 cls_w_l1_regular_loss_weight=1e-3,
                 reg_cls_mutex=False,
                 cls_reg_mutex_thresh=0.1,
                 key_channels_ratio=None,
                 amsoftmax_m=None,
                 use_cos=False,
                 cos_scale=20,
                 bg_fg_mutex=False,
                 loss_bg_fg_mutex_weight=1.0,
                 multi_rois=False,
                 roi_offsets=None,
                 add_entropy_regular=False,
                 entropy_regular_weight=1.,
                 add_cos_center_loss=False,
                 triplet_margin=None,
                 triplet_loss_weight=1.,
                 add_soft_cos_center_loss=False,
                 soft_center_thresh=0.7,
                 cos_center_loss_weight=0.1,
                 add_cos=False,
                 cls_w_att=False,
                 detach=True,
                 freeze_att=False,
                 alpha=1.,
                 use_channel_attention=False,
                 channel_attention_loss_weight=1.,
                 freeze_channel_attention=False,
                 base_ids=None,
                 novel_ids=None,
                 grad_scale=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(FSBBoxHead, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        self.use_channel_attention = use_channel_attention
        self.channel_attention_loss_weight = channel_attention_loss_weight
        if use_channel_attention:
            self.cls_channel_attention = ChannelAttention(in_channels, freeze=freeze_channel_attention)
            self.reg_channel_attention = ChannelAttention(in_channels, freeze=freeze_channel_attention)

        self.spicial_att = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        if freeze_att:
            self.spicial_att.eval()
            for p in self.spicial_att.parameters():
                p.requires_grad = False
        self.use_cos = use_cos
        self.cos_scale = cos_scale
        self.add_cos_center_loss = add_cos_center_loss
        self.cos_center_loss_weight = cos_center_loss_weight
        if self.with_cls:
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            self.fc_cls = nn.Linear(in_channels, fc_cls_channels, bias=not use_cos)
        if self.with_reg:
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            out_dim_reg = 4 if reg_class_agnostic else 4 * fc_cls_channels
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        self.dropout_ratio = dropout_ratio
        self.base_ids = base_ids
        self.novel_ids = novel_ids
        self.grad_scale = grad_scale
        self.alpha = alpha
        self.detach = detach
        self.cls_w_att = cls_w_att
        self.add_cos = add_cos
        self.cls_w_l1_regular = cls_w_l1_regular
        self.cls_w_l1_regular_loss_weight = cls_w_l1_regular_loss_weight
        self.reg_cls_mutex = reg_cls_mutex
        self.cls_reg_mutex_thresh = cls_reg_mutex_thresh
        self.add_center_loss = add_center_loss
        self.center_loss_weight = center_loss_weight
        self.key_channels_ratio = key_channels_ratio
        self.amsoftmax_m = amsoftmax_m

        self.triplet_margin = triplet_margin
        self.add_soft_cos_center_loss = add_soft_cos_center_loss
        self.soft_center_thresh = soft_center_thresh
        self.triplet_loss_weight = triplet_loss_weight
        self.add_entropy_regular = add_entropy_regular
        self.entropy_regular_weight = entropy_regular_weight
        self.multi_rois = multi_rois
        self.roi_offsets = roi_offsets

        self.bg_fg_mutex = bg_fg_mutex
        self.loss_bg_fg_mutex_weight = loss_bg_fg_mutex_weight

        if add_center_loss:
            num_fg = (len(base_ids)+len(novel_ids)) if base_ids is not None else num_classes - 1
            self.centers = nn.Parameter(torch.zeros(num_fg, in_channels), requires_grad=True)
        if add_cos:
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            self.cos_cls = nn.Linear(in_channels, fc_cls_channels, bias=False)
        self.cls_feat_att = nn.Parameter(torch.ones(1, in_channels), requires_grad=False)

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        nn.init.normal_(self.spicial_att.weight, 0, 0.01)
        nn.init.constant_(self.spicial_att.bias, 0)

        if self.with_cls and not self.use_cos:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x, extra=None):
        if self.training and (self.grad_scale is not None):
            x = scale_tensor_gard(x, self.grad_scale)
        bs, c, w, h = x.shape

        x_avg_pool = self.avg_pool(x).reshape(bs, -1)

        in_feat = x.detach() if self.detach else x
        spicial_att = self.spicial_att(in_feat).reshape(bs, 1 , -1).softmax(-1)
        # show_feature(spicial_att[None, None, :, 0, :], use_sigmoid=False, bar_scope=(0, 1))
        x = x.reshape(bs, c, -1)
        x = x * spicial_att
        x = x.sum(-1)

        x = self.alpha * x + (1 - self.alpha) * x_avg_pool

        if self.use_channel_attention:
            cls_feat = self.cls_channel_attention(x, dim=1)
            reg_feat = self.reg_channel_attention(x, dim=1)
        else:
            cls_feat = x
            reg_feat = x

        if self.dropout_ratio is not None:
            cls_feat = F.dropout(cls_feat, self.dropout_ratio, training=self.training)

        if self.cls_w_att:
            if self.base_ids is not None:
                out_channels = []
                w_att = torch.zeros_like(self.fc_cls.weight.data, device=x.device)
                for id in self.base_ids:
                    out_channels.append(id+1)
                w_att[out_channels, :] = self.fc_cls.weight.data[out_channels, :]
                w_att.abs_()
                w_att = w_att / w_att.sum(0, keepdim=True)
                out_channels = [0]
                for id in self.novel_ids:
                    out_channels.append(id+1)
                w_att[out_channels, :] = 1.
                self.fc_cls.weight.data = self.fc_cls.weight.data * w_att
            else:
                w_att = torch.zeros_like(self.fc_cls.weight.data, device=x.device)
                w_att[1:, :] = self.fc_cls.weight.data[1:, :]
                w_att.abs_()
                w_att = w_att / w_att.sum(0, keepdim=True)
                w_att[0, :] = 1.
                self.fc_cls.weight.data = self.fc_cls.weight.data * w_att

        # if self.reg_cls_mutex:
        #     reg_w = self.fc_reg.weight
        #     reg_w_max = reg_w.abs().max(dim=0, keepdim=True)[0]
        #     mutex_mask = reg_w_max < self.cls_reg_mutex_thresh
        #     mutex_mask = mutex_mask.float()
        if self.reg_cls_mutex:
            reg_w = self.fc_reg.weight
            reg_w_max = reg_w.abs().max(dim=1, keepdim=True)[0]
            mutex_mask = (reg_w / reg_w_max).max(dim=0, keepdim=True)[0]
            mutex_mask = torch.sub(1., mutex_mask)

        if self.multi_rois:
            if self.training:
                with torch.no_grad():
                    multi_roi_feat, multi_roi_label = extra
                    bs, c, w, h = multi_roi_feat.shape
                    spicial_att = self.spicial_att(multi_roi_feat).reshape(bs, 1, -1).softmax(-1)
                    # show_feature(spicial_att[None, None, :, 0, :], use_sigmoid=False, bar_scope=(0, 1))
                    multi_roi_feat = multi_roi_feat.reshape(bs, c, -1)
                    multi_roi_feat = multi_roi_feat * spicial_att
                    multi_roi_feat = multi_roi_feat.sum(-1)

                    multi_roi_feat = multi_roi_feat.reshape(-1, len(self.roi_offsets), self.in_channels)
                    gt_roi_feat = multi_roi_feat[:, 0:1, :]
                    other_roi_feat = multi_roi_feat[:, 1:, :]
                    sub_feat = other_roi_feat - gt_roi_feat
                    sub_feat_relu = F.relu(sub_feat)
                    sub_feat_relu_max = sub_feat_relu.max(dim=-1, keepdim=True)[0] + 1e-5
                    cls_att = torch.sub(1, sub_feat_relu / sub_feat_relu_max).mean(1).mean(0, keepdim=True)
                    self.cls_feat_att.data = self.cls_feat_att.data + 0.01 * (cls_att - self.cls_feat_att.data)
            cls_feat = cls_feat * self.cls_feat_att

        if self.use_cos:
            fg_w_norm = F.normalize(self.fc_cls.weight[1:, :], p=2, dim=1)
            if self.reg_cls_mutex:
                cls_feat_norm = F.normalize(cls_feat * mutex_mask, p=2, dim=1)
            else:
                cls_feat_norm = F.normalize(cls_feat, p=2, dim=1)
            # cls_feat_norm = F.normalize(cls_feat, p=2, dim=1)
            fg_w_norm_ex = fg_w_norm[None, :, :].expand(cls_feat.size(0), -1, -1)
            cls_feat_norm_ex = cls_feat_norm[:, None, :].expand_as(fg_w_norm_ex)
            fg_cls_score = (fg_w_norm_ex * cls_feat_norm_ex).sum(-1) * self.cos_scale
            bg_cls_score = (self.fc_cls.weight[0, :][None, :].expand_as(cls_feat) * cls_feat).sum(-1, keepdim=True)
            cls_score = torch.cat([bg_cls_score, fg_cls_score], dim=1)
        else:
            if self.key_channels_ratio is not None:
                num_key_chaannels = int(self.in_channels * self.key_channels_ratio)
                thresh = cls_feat.sort(dim=1, descending=True)[0][:, num_key_chaannels][:, None].expand_as(cls_feat)
                inds = cls_feat < thresh
                cls_feat[inds] = cls_feat[inds] * 0.

            cls_score = self.fc_cls(cls_feat) if self.with_cls else None
        bbox_pred = self.fc_reg(reg_feat) if self.with_reg else None

        if self.add_cos:
            cls_feat_norm = F.normalize(cls_feat, p=2, dim=1)
            self.cos_cls.weight.data = F.normalize(self.cos_cls.weight.data, p=2, dim=1)
            cls_score_cos = self.cos_cls(cls_feat_norm) * self.cos_scale

        if self.base_ids is not None:
            out_channels = [0]
            for id in self.base_ids:
                out_channels.append(id+1)
            if cls_score is not None:
                cls_score = cls_score[:, out_channels]
            if self.add_cos:
                cls_score_cos = cls_score_cos[:, out_channels]
            if not self.reg_class_agnostic and bbox_pred is not None:
                bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1, 4)[:, out_channels, :].reshape(bbox_pred.size(0), -1)

        if self.add_center_loss:
            if self.base_ids is not None:
                centers = self.centers[self.base_ids, :]
            else:
                centers = self.centers
            cls_feat_ex = cls_feat[:, None, :].expand(-1, centers.size(0), -1)
            centers_ex = centers[None, :, :].expand_as(cls_feat_ex)
            dis_center_l2 = torch.sqrt(((cls_feat_ex - centers_ex)**2).sum(-1))

        if self.add_center_loss:
            if self.training:
                return cls_score, bbox_pred, dis_center_l2
            else:
                return cls_score, bbox_pred
        if self.add_cos:
            if self.training:
                return (cls_score, cls_score_cos), bbox_pred, None
            else:
                return (cls_score, cls_score_cos), bbox_pred

        if self.training:
            return cls_score, bbox_pred, None
        else:
            # cls_score = torch.zeros_like(cls_score)
            # cls_score[:, 1] = 5.
            # bbox_pred = torch.zeros_like(bbox_pred)
            return cls_score, bbox_pred

    def finetune_init(self, x):
        bs, c, w, h = x.shape
        spicial_att = self.spicial_att(x).reshape(bs, 1, -1).softmax(-1)
        x = x.reshape(bs, c, -1)
        x = x * spicial_att
        x_vector = x.sum(-1)
        return x_vector

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'extra'))
    def loss(self,
             cls_score,
             bbox_pred,
             extra,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if self.add_cos:
            cls_score, cls_score_cos = cls_score
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                if self.amsoftmax_m is not None:
                    labels_oh = F.one_hot(labels.flatten(0), num_classes=self.num_classes).byte()
                    labels_oh[:, 0] = 0
                    cls_score[labels_oh] -= self.amsoftmax_m * self.cos_scale
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if self.add_cos:
                    losses['loss_cls_cos'] = self.loss_cls(
                        cls_score_cos,
                        labels,
                        label_weights,
                        avg_factor=avg_factor,
                        reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if pos_inds.any():
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)

        if self.bg_fg_mutex:
            pass

        if self.use_cos:
            if (self.add_cos_center_loss or self.add_soft_cos_center_loss):
                pos_inds = labels > 0
                labels_pos = labels[pos_inds]
                labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=self.num_classes).byte()[:, 1:]
                cls_score_pos = cls_score[:, 1:][pos_inds]
                cos_sim_pos = cls_score_pos[labels_pos_one_hot] / self.cos_scale
                if self.add_cos_center_loss:
                    losses['loss_cos_center'] = torch.sub(1, cos_sim_pos).mean() * self.cos_center_loss_weight
                if self.add_soft_cos_center_loss:
                    losses['loss_soft_cos_center'] = F.relu(self.soft_center_thresh - cos_sim_pos).mean() * self.cos_center_loss_weight
            if self.triplet_margin is not None:
                pos_inds = labels > 0
                labels_pos = labels[pos_inds]
                labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=self.num_classes).byte()[:, 1:]
                cls_score_pos = cls_score[:, 1:][pos_inds]
                cos_sim_pos = cls_score_pos / self.cos_scale
                bs = cos_sim_pos.size(0)
                cos_sim_pos_gt = cos_sim_pos[labels_pos_one_hot]
                cos_sim_pos_other = cos_sim_pos[torch.sub(1, labels_pos_one_hot)].reshape(bs, -1)
                cos_sim_pos_other_min = cos_sim_pos_other.max(dim=-1)[0]
                losses['loss_triplet'] = F.relu(cos_sim_pos_other_min - cos_sim_pos_gt + self.triplet_margin).mean() * self.triplet_loss_weight
                # losses_triplet = F.relu(cos_sim_pos_other_min - cos_sim_pos_gt + self.triplet_margin)
                # num_avg = int((losses_triplet > 0).sum())
                # num_avg = max(num_avg, 1)
                # losses['loss_triplet'] = losses_triplet / num_avg * self.triplet_loss_weight
        if self.add_entropy_regular:
            if self.use_cos:
                cos_sim = F.relu(cls_score[:, 1:] / self.cos_scale)
                cos_sim_ln = torch.log(cos_sim + 1e-5)
                losses['loss_entropy_regular'] = (-cos_sim * cos_sim_ln).sum(-1).mean() * self.entropy_regular_weight
                # cls_score_softmax = cls_score.softmax(-1)
                # cls_score_softmax_ln = torch.log(cls_score_softmax)
                # losses['loss_entropy_regular'] = (-cls_score_softmax * cls_score_softmax_ln).sum(-1).mean() * self.entropy_regular_weight


        if self.add_center_loss:
            pos_inds = labels > 0
            labels_pos = labels[pos_inds]
            dis_center_l2_pos = extra[pos_inds]
            labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=self.num_classes).byte()[:, 1:]
            losses['loss_center'] = dis_center_l2_pos[labels_pos_one_hot].mean() * self.center_loss_weight

        if self.cls_w_l1_regular:
            if self.base_ids is not None:
                output_channels = [i + 1 for i in self.base_ids]
                losses['loss_cls_w_l1'] = self.fc_cls.weight[output_channels, :].abs().sum() * self.cls_w_l1_regular_loss_weight
            else:
                losses['loss_cls_w_l1'] = self.fc_cls.weight.abs().sum() * self.cls_w_l1_regular_loss_weight
        if self.use_channel_attention:
            losses['loss_CA_cls'] = self.cls_channel_attention.channel_attention.abs().sum() * self.channel_attention_loss_weight
            losses['loss_CA_reg'] = self.reg_channel_attention.channel_attention.abs().sum() * self.channel_attention_loss_weight
        return losses

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if self.add_cos:
            cls_score, cls_score_cos = cls_score
            if isinstance(cls_score_cos, list):
                cls_score_cos = sum(cls_score_cos) / float(len(cls_score_cos))
            cls_score_cos = F.softmax(cls_score_cos, dim=1) if cls_score_cos is not None else None

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if self.add_cos:
            scores = scores * 0.5 + cls_score_cos * 0.5

        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = torch.from_numpy(scale_factor).to(bboxes.device)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image. The first column is
                the image id and the next 4 columns are x1, y1, x2, y2.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.

        Example:
            >>> # xdoctest: +REQUIRES(module:kwarray)
            >>> import kwarray
            >>> import numpy as np
            >>> from mmdet.core.bbox.demodata import random_boxes
            >>> self = BBoxHead(reg_class_agnostic=True)
            >>> n_roi = 2
            >>> n_img = 4
            >>> scale = 512
            >>> rng = np.random.RandomState(0)
            >>> img_metas = [{'img_shape': (scale, scale)}
            ...              for _ in range(n_img)]
            >>> # Create rois in the expected format
            >>> roi_boxes = random_boxes(n_roi, scale=scale, rng=rng)
            >>> img_ids = torch.randint(0, n_img, (n_roi,))
            >>> img_ids = img_ids.float()
            >>> rois = torch.cat([img_ids[:, None], roi_boxes], dim=1)
            >>> # Create other args
            >>> labels = torch.randint(0, 2, (n_roi,)).long()
            >>> bbox_preds = random_boxes(n_roi, scale=scale, rng=rng)
            >>> # For each image, pretend random positive boxes are gts
            >>> is_label_pos = (labels.numpy() > 0).astype(np.int)
            >>> lbl_per_img = kwarray.group_items(is_label_pos,
            ...                                   img_ids.numpy())
            >>> pos_per_img = [sum(lbl_per_img.get(gid, []))
            ...                for gid in range(n_img)]
            >>> pos_is_gts = [
            >>>     torch.randint(0, 2, (npos,)).byte().sort(
            >>>         descending=True)[0]
            >>>     for npos in pos_per_img
            >>> ]
            >>> bboxes_list = self.refine_bboxes(rois, labels, bbox_preds,
            >>>                    pos_is_gts, img_metas)
            >>> print(bboxes_list)
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze(dim=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)

            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds.type(torch.bool)])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5, repr(rois.shape)

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
