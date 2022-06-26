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
from mmdet.core.bbox.geometry import bbox_overlaps

from mmdet.models.subnets import MLP
from mmdet.utils.show_feature import show_feature

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

@HEADS.register_module
class MetaFSCosBBoxHead(nn.Module):
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
                 base_ids=None,
                 novel_ids=None,
                 grad_scale=None,
                 cos_scale=3,
                 spicial_att=False,
                 triplet_margin=None,
                 triplet_loss_weight=1.0,
                 neg_cls=False,
                 hn_thresh=(0.1, 0.5),
                 neg_beta=0.3,
                 use_tanh=False,
                 dropout=False,
                 dropout_p=0.8,
                 current_epoch=0,
                 epoch_thresh=4,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(MetaFSCosBBoxHead, self).__init__()
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

        # self.loss_cls = build_loss(loss_cls)
        self.loss_cls = build_loss(dict(
            type='FocalLoss',
            use_sigmoid=not use_tanh,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0))
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        if self.with_cls:
            fc_cls_channels = (len(base_ids)+len(novel_ids)) if base_ids is not None else num_classes - 1
            self.fc_cls = nn.Linear(in_channels, fc_cls_channels, bias=False)
            self.meta_representatives = nn.Parameter(torch.zeros(fc_cls_channels, in_channels), requires_grad=False)
            self.meta_representatives_counter = nn.Parameter(torch.zeros(fc_cls_channels), requires_grad=False)

        if self.with_reg:
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            out_dim_reg = 4 if reg_class_agnostic else 4 * fc_cls_channels
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        self.base_ids = base_ids
        self.grad_scale = grad_scale

        self.cos_scale = cos_scale

        self.spicial_att = spicial_att
        if spicial_att:
            self.spicial_att_conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

        self.triplet_margin = triplet_margin
        self.triplet_loss_weight = triplet_loss_weight

        self.neg_cls = neg_cls
        self.hn_thresh = hn_thresh
        self.neg_beta = neg_beta
        if neg_cls:
            fc_cls_channels = (len(base_ids)+len(novel_ids)) if base_ids is not None else num_classes - 1
            self.fc_neg_cls = nn.Linear(in_channels, fc_cls_channels, bias=False)

        self.use_tanh = use_tanh

        self.dropout = dropout
        self.dropout_p = dropout_p

        self.current_epoch = current_epoch
        self.epoch_thresh = epoch_thresh

    def init_weights(self):
        if self.neg_cls:
            nn.init.normal_(self.fc_neg_cls.weight, 0, 0.01)
        if self.spicial_att:
            nn.init.normal_(self.spicial_att_conv.weight, 0, 0.01)
            nn.init.constant_(self.spicial_att_conv.bias, 0)
        # conv layers are already initialized by ConvModule
        # if self.with_cls:
        #     nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            # nn.init.constant_(self.fc_cls.bias, 0)

        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x, extra=None, pre_test=False):
        if self.training and (self.grad_scale is not None):
            x = scale_tensor_gard(x, self.grad_scale)

        extra_return = None

        if extra is not None:
            instance_propotypes, support_instance_labels = extra
            instance_propotypes = self.avg_pool(instance_propotypes)
            instance_propotypes = instance_propotypes.view(instance_propotypes.size(0), -1)
            # instance_propotypes_norm = F.normalize(instance_propotypes, p=2, dim=1)
            # print(instance_propotypes.shape)
            # print(support_instance_labels)

            if pre_test:
                for l in support_instance_labels:
                    self.meta_representatives[l-1] += instance_propotypes
                    self.meta_representatives_counter[l-1] += 1
                    return

            extra_return = support_instance_labels
        else:
            instance_propotypes = self.meta_representatives

        bs, c, w, h = x.shape
        if self.spicial_att:
            spicial_att = self.spicial_att_conv(x).reshape(bs, 1 , -1).softmax(-1)
            # show_feature(spicial_att[None, None, :, 0, :], use_sigmoid=False, bar_scope=(0, 1))
            x = x.reshape(bs, c, -1)
            x = x * spicial_att
            x = x.sum(-1)
        else:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        # cls_score = self.fc_cls(x) if self.with_cls else None
        cls_feat = x
        reg_feat = x

        fg_w = self.fc_cls.weight

        if self.base_ids is not None:
            fg_w = fg_w[self.base_ids, :]

        if extra is not None:
            fg_w = fg_w[support_instance_labels-1, :]

        fg_w_ex = fg_w[None, :, :].expand(cls_feat.size(0), -1, -1)
        cls_feat_ex = cls_feat[:, None, :].expand_as(fg_w_ex)
        cls_att = instance_propotypes.sigmoid()[None, :, :].expand_as(cls_feat_ex)
        cls_feat_ex = cls_feat_ex * 0.5 + cls_att * cls_feat_ex
        cos_sim = F.cosine_similarity(fg_w_ex, cls_feat_ex, dim=2)

        # instance_propotypes_norm_ex = instance_propotypes_norm[None, :, :].expand(cls_feat.size(0), -1, -1)
        # cls_feat_norm_ex = cls_feat_norm[:, None, :].expand_as(instance_propotypes_norm_ex)
        # cos_sim = (instance_propotypes_norm_ex * cls_feat_norm_ex).sum(-1)
        # print(cos_sim.max())
        # print(cos_sim.min())
        # cls_score = cos_sim * self.cos_scale

        if self.neg_cls:
            neg_fg_w_norm = F.normalize(self.fc_neg_cls.weight, p=2, dim=1)
            neg_fg_w_norm_ex = neg_fg_w_norm[None, :, :].expand(cls_feat.size(0), -1, -1)
            # cos_sim_neg = (neg_fg_w_norm_ex * cls_feat_norm_ex).sum(-1)
            cos_sim_neg = 1
            cls_score = torch.exp(-(torch.sub(1, cos_sim - self.neg_beta * cos_sim_neg))**2 * self.cos_scale)
        elif self.use_tanh:
            cls_score = torch.exp(-(torch.sub(1, cos_sim))**2 * self.cos_scale)
            cls_score = (cls_score * self.cos_scale).tanh()
        else:
            cls_score = torch.exp(-(torch.sub(1, cos_sim))**2 * self.cos_scale)

        if not self.use_tanh:
            cls_score = inverse_sigmoid(cls_score)
        # cls_score = self.fc_cls(cls_feat)
        bbox_pred = self.fc_reg(reg_feat) if self.with_reg else None
        if self.base_ids is not None:
            out_channels = [0]
            for id in self.base_ids:
                out_channels.append(id+1)
            # if cls_score is not None and extra is None:
            #     cls_score = cls_score[:, self.base_ids]
            if not self.reg_class_agnostic and bbox_pred is not None:
                bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1, 4)[:, out_channels, :].reshape(bbox_pred.size(0), -1)

        # if self.img_classification and not self.training:
        #     cls_score = cls_score_img.sigmoid() * cls_score

        if self.triplet_margin is not None:
            if self.base_ids is not None:
                cos_sim = cos_sim[:, self.base_ids]
            extra_return = cos_sim

        if self.neg_cls:
            if self.base_ids is not None:
                cos_sim_neg = cos_sim_neg[:, self.base_ids]
            extra_return = cos_sim_neg

        # cls_score = torch.zeros_like(cls_score)
        # cls_score[:, 1] = 5.
        # bbox_pred = torch.zeros_like(bbox_pred)
        if self.training:
            return cls_score, bbox_pred, extra_return
        else:
            return cls_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        labels, label_weights, bbox_targets, bbox_weights = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds,
            concat=False)
        if self.neg_cls:
            num_img = len(sampling_results)
            for i in range(num_img):
                num_pos = pos_proposals[i].size(0)
                overlaps = bbox_overlaps(gt_bboxes[i], neg_proposals[i])
                max_overlaps, argmax_overlaps = overlaps.max(dim=0)
                hn_inds = (max_overlaps >= self.hn_thresh[0]) & (max_overlaps < self.hn_thresh[1])
                neg_labels = labels[i][num_pos:]
                neg_labels[hn_inds] = -gt_labels[i][argmax_overlaps[hn_inds]]
                labels[i] = torch.cat([labels[i][:num_pos], neg_labels], dim=0)

        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
        return (labels, label_weights, bbox_targets, bbox_weights)

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
        # print('loss()')
        # print(labels)
        # print(extra)
        losses = dict()
        for i, l in enumerate(extra):
            labels[labels==l] = i + 1
        # print(labels)

        if self.neg_cls:
            cos_sim_neg = extra
            neg_labels = labels.clone()
            labels[labels < 0] = 0
            neg_labels[neg_labels > 0] = 0
            neg_labels[neg_labels < 0] = -neg_labels[neg_labels < 0]
            num_neg_pos = max(torch.sum(neg_labels > 0).float().item(), 1.)
            if num_neg_pos == 0:
                losses['loss_neg_cls'] = torch.zeros(1).to(cls_score.device)
            else:
                neg_cls_score = torch.exp(-(torch.sub(1, cos_sim_neg))**2 * self.cos_scale)
                neg_cls_score = inverse_sigmoid(neg_cls_score)
                label_weights[neg_labels > 0] = 1.
                losses['loss_neg_cls'] = self.loss_cls(neg_cls_score, neg_labels, label_weights, avg_factor=num_neg_pos)

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            num_pos = max(torch.sum(labels > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                # losses['loss_cls'] = self.loss_cls(
                #     cls_score,
                #     labels,
                #     label_weights,
                #     avg_factor=avg_factor,
                #     reduction_override=reduction_override)
                # labels_oh = F.one_hot(labels, num_classes=self.num_classes)[:, 1:]
                # loss_bce = F.binary_cross_entropy(cls_score, labels_oh.type_as(cls_score), reduction='none')
                # losses['loss_cls'] = (loss_bce.mean(1) * label_weights).sum(0) / num_pos
                if self.use_tanh:
                    labels_oh = F.one_hot(labels, num_classes=self.num_classes)[:, 1:]
                    losses['loss_cls'] = self.loss_cls(cls_score, labels_oh, label_weights, avg_factor=num_pos)
                else:
                    losses['loss_cls'] = self.loss_cls(cls_score, labels, label_weights, avg_factor=num_pos)
                cls_score_bg = torch.sub(1, cls_score.detach().max(1, keepdim=True)[0])
                losses['acc'] = accuracy(torch.cat([cls_score_bg, cls_score], dim=1), labels)
                if (labels > 0).sum() > 0:
                    losses['fg_acc'] = accuracy(torch.cat([cls_score_bg, cls_score], dim=1)[labels > 0], labels[labels > 0])
                else:
                    losses['fg_acc'] = 100.

        if self.triplet_margin is not None:
            cos_sim = extra
            pos_inds = labels > 0
            labels_pos = labels[pos_inds]
            labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=self.num_classes).byte()[:, 1:]
            cos_sim_pos = cos_sim[pos_inds]
            bs = cos_sim_pos.size(0)
            cos_sim_pos_gt = cos_sim_pos[labels_pos_one_hot]
            cos_sim_pos_other = cos_sim_pos[torch.sub(1, labels_pos_one_hot)].reshape(bs, -1)
            cos_sim_pos_other_max = cos_sim_pos_other.max(dim=-1)[0]
            losses['loss_triplet'] = F.relu(
                cos_sim_pos_other_max - cos_sim_pos_gt + self.triplet_margin).mean() * self.triplet_loss_weight

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
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        if not self.use_tanh:
            cls_score = cls_score.sigmoid()
        pad = torch.zeros(cls_score.size(0), 1).to(cls_score.device)
        scores = torch.cat([pad, cls_score], dim=1)
        # scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

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
