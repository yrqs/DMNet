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

from mmdet.utils.print_score import print_voc_score

@HEADS.register_module
class BBoxHead(nn.Module):
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
                 neg_cls=False,
                 hn_thresh=(0.2, 0.4),
                 enhance_hn=False,
                 enhance_hn_thresh=0.1,
                 enhance_hn_weight=1.0,
                 img_classification=False,
                 global_attention=False,
                 global_info=False,
                 dropout_p=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHead, self).__init__()
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
        if self.with_cls:
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            if global_info:
                self.fc_cls = nn.Linear(in_channels * 2, fc_cls_channels)
            else:
                self.fc_cls = nn.Linear(in_channels, fc_cls_channels)
        if self.with_reg:
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            out_dim_reg = 4 if reg_class_agnostic else 4 * fc_cls_channels
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        self.base_ids = base_ids
        self.grad_scale = grad_scale

        self.enhance_hn = enhance_hn
        self.enhance_hn_thresh = enhance_hn_thresh
        self.enhance_hn_weight = enhance_hn_weight

        self.img_classification = img_classification
        if img_classification:
            img_cls_channels = (len(base_ids)+len(novel_ids)) if base_ids is not None else num_classes - 1
            self.img_cls = nn.Linear(in_channels, img_cls_channels)

        self.global_attention = global_attention
        if global_attention:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.global_info = global_info
        if global_info:
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.neg_cls = neg_cls
        self.hn_thresh = hn_thresh
        if neg_cls:
            fc_cls_channels = (len(base_ids)+len(novel_ids)) if base_ids is not None else num_classes - 1
            self.neg_fc_cls = nn.Linear(in_channels, fc_cls_channels)

        self.cls_fc_w_mask = nn.Parameter(torch.ones_like(self.fc_cls.weight), requires_grad=False)
        # for p in self.img_cls.parameters():
        #     p.requires_grad = False
        self.dropout_p = dropout_p

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.neg_cls:
            nn.init.normal_(self.neg_fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.neg_fc_cls.bias, 0)
        if self.img_classification:
            nn.init.normal_(self.img_cls.weight, 0, 0.01)
            nn.init.constant_(self.img_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x, extra=None):
        if self.training and (self.grad_scale is not None):
            x = scale_tensor_gard(x, self.grad_scale)
        extra_return = None

        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_feat = x
        reg_feat = x
        # self.fc_cls.weight.data = self.fc_cls.weight.data * self.cls_fc_w_mask
        if self.global_attention:
            x_global = extra
            if self.training and (self.grad_scale is not None):
                x_global = scale_tensor_gard(x_global, self.grad_scale)
            x_global_avg = self.global_avg_pool(x_global).flatten(1)
            x_global_max = self.global_max_pool(x_global).flatten(1)
            bs = x_global_avg.shape[0]
            cls_feat = cls_feat.reshape(bs, -1, self.in_channels)
            att = (x_global_avg + 0.2 * x_global_max).tanh().detach()
            cls_feat = cls_feat * att[:, None, :].expand_as(cls_feat)
            cls_feat = cls_feat.reshape(-1, self.in_channels)
        if self.global_info:
            x_global = extra
            if self.training and (self.grad_scale is not None):
                x_global = scale_tensor_gard(x_global, self.grad_scale)
            x_global_avg = self.global_avg_pool(x_global).flatten(1).expand_as(cls_feat)
            # x_global_max = self.global_max_pool(x_global).flatten(1).expand_as(cls_feat)s
            cls_feat = torch.cat([cls_feat, x_global_avg], dim=1)

        if self.dropout_p is not None:
            cls_feat = F.dropout(cls_feat, self.dropout_p, self.training)

        cls_score = self.fc_cls(cls_feat) if self.with_cls else None
        bbox_pred = self.fc_reg(reg_feat) if self.with_reg else None
        if self.base_ids is not None:
            out_channels = [0]
            for id in self.base_ids:
                out_channels.append(id+1)
            if cls_score is not None:
                cls_score = cls_score[:, out_channels]
            if not self.reg_class_agnostic and bbox_pred is not None:
                bbox_pred = bbox_pred.reshape(bbox_pred.size(0), -1, 4)[:, out_channels, :].reshape(bbox_pred.size(0), -1)

        if self.neg_cls:
            neg_cls_score = self.neg_fc_cls(cls_feat)
            if self.base_ids is not None:
                neg_cls_score = neg_cls_score[:, self.base_ids]
            if self.training:
                cls_score = torch.cat([cls_score, neg_cls_score], dim=1)
            else:
                bg_score = cls_score[:, :1]
                bg_score = torch.cat([bg_score, neg_cls_score], dim=1).max(1, keepdim=True)[0]
                cls_score = torch.cat([bg_score, cls_score[:, 1:]], dim=1)

        if self.img_classification:
            img_labels = extra
            img_cls_score = self.img_cls(x)
            if self.base_ids is not None:
                img_cls_score = img_cls_score[:, self.base_ids]
            img_cls_score = img_cls_score.softmax(0) * img_cls_score.softmax(1)
            img_cls_score = img_cls_score.sum(0, keepdim=True)
            # img_cls_score = cls_score.softmax(0) * cls_score.softmax(1)
            # print(img_cls_score.shape)
            # img_cls_score = img_cls_score.sum(0, keepdim=True)[:, 1:]
            # print(img_cls_score.shape)
            extra_return = (img_cls_score, img_labels)
        # cls_score = torch.zeros_like(cls_score)
        # cls_score[:, 1] = 5.
        # bbox_pred = torch.zeros_like(bbox_pred)
        if self.training:
            return cls_score, bbox_pred, extra_return
        else:
            if self.img_classification:
                return (cls_score, img_cls_score), bbox_pred
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
        losses = dict()
        if self.img_classification:
            img_cls_score, img_labels = extra
            img_label_bce = []
            print(img_labels)
            for l in img_labels:
                if l.shape[0] > 0:
                    l_oh = F.one_hot(l, num_classes=self.num_classes)[:, 1:]
                    img_label_bce.append(l_oh.sum(0, keepdim=True))
            # print(img_cls_score.shape)
            # print(torch.cat(img_label_bce, dim=0).float().shape)
            if len(img_label_bce) > 0:
                img_label_bce = torch.cat(img_label_bce, dim=0).float()
                losses['loss_img_cls'] = F.binary_cross_entropy(img_cls_score, img_label_bce)
            else:
                losses['loss_img_cls'] = torch.zeros(1).to(cls_score.device)

        if self.neg_cls:
            hn_inds = labels < 0
            labels[hn_inds] = -labels[hn_inds] + self.num_classes - 1

        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)
                losses['fg_acc'] = accuracy(cls_score[labels > 0], labels[labels > 0])

                if self.enhance_hn:
                    cls_score_softmax = F.softmax(cls_score)
                    predict_cls = cls_score_softmax.max(dim=1)[1]
                    hn_ind = predict_cls != labels
                    hn_cls_score_softmax = cls_score_softmax[hn_ind]
                    hn_labels = predict_cls[hn_ind]
                    # print(hn_labels)
                    # print(labels[hn_ind])
                    hn_labels_oh = F.one_hot(hn_labels, num_classes=self.num_classes).byte()
                    # print('============1==============')
                    # print(hn_cls_score_softmax.shape)
                    # print(hn_labels_oh.shape)
                    hn_samples = hn_cls_score_softmax[hn_labels_oh]
                    # print(hn_samples.shape)
                    hn_samples = hn_samples[hn_samples >= self.enhance_hn_thresh]
                    if hn_samples.shape[0] > 0:
                    # print(hn_samples.shape)
                        losses['loss_enhance_hn'] = hn_samples.mean() * self.enhance_hn_weight
                    else:
                        losses['loss_enhance_hn'] = torch.zeros(1).to(cls_score.device)

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
        if self.img_classification:
            cls_score, img_cls_score = cls_score
            # img_cls_score = img_cls_score.sigmoid()
            assert img_cls_score.shape[0] == 1

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        # if self.img_classification:
        #     scores[:, 1:] = scores[:, 1:] * img_cls_score.expand_as(scores[:, 1:])

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
