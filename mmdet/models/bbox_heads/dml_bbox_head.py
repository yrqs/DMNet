import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from mmcv.cnn import normal_init, constant_init
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from mmdet.ops.scale_grad import scale_tensor_gard

class DMLHead(nn.Module):
    def __init__(self,
                 emb_module,
                 cls_num,
                 emb_channels,
                 num_modes,
                 sigma,
                 cls_norm,
                 base_ids=None,
                 novel_ids=None,
                 freeze=False):
        assert num_modes==1
        super().__init__()
        self.num_reps = len(base_ids) + len(novel_ids) if base_ids is not None else cls_num
        self.num_reps += 1
        self.cls_num = cls_num
        self.num_modes = num_modes
        self.sigma = sigma
        self.emb_module = emb_module
        self.emb_channels = emb_channels
        self.cls_norm = cls_norm
        self.rep_fc = nn.Linear(1, self.num_reps * num_modes * emb_channels[-1])
        self.representations = nn.Parameter(
            torch.zeros(self.num_reps, self.num_modes, self.emb_channels[-1]),
            requires_grad=False)
        normal_init(self.rep_fc, std=0.01)

        self.base_ids = base_ids
        self.novel_ids = novel_ids
        if freeze:
            for c in [self.neg_offset_fc]:
                for p in c.parameters():
                    p.requires_grad = False

    def forward(self, x, save_outs=False):
        emb_vectors = x
        if self.emb_module is not None:
            emb_vectors = self.emb_module(x)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        if self.training:
            reps = self.rep_fc(torch.tensor(1.0).to(x.device).unsqueeze(0)).squeeze(0)
            reps  = reps.view(self.num_reps, self.num_modes, self.emb_channels[-1])
            reps = F.normalize(reps, p=2, dim=2)
            self.representations.data = reps.detach()
        else:
            reps = self.representations.detach()

        if self.base_ids is not None:
            ids = [0] + [i + 1 for i in self.base_ids]
            reps = reps[ids, :, :]

        reps_ex = reps[None, :, :, :]
        emb_vectors_ex = emb_vectors[:, None, None, :]

        distances = torch.sqrt(((emb_vectors_ex - reps_ex)**2).sum(-1))

        # probs_ori = torch.exp(-(distances)**2/(2.0*self.sigma**2))
        # probs_ori = probs_ori.max(dim=2)[0]
        probs = torch.exp(-(distances)**2/(2.0*self.sigma**2))
        # bg_probs = torch.sub(1, probs.max(dim=1, keepdim=True)[0])
        # probs = torch.cat([bg_probs, probs], dim=1)
        if self.cls_norm:
            probs_sumj = probs.sum(2)
            probs_sumij = probs_sumj.sum(1, keepdim=True)
            cls_score = probs_sumj / probs_sumij
        else:
            cls_score = probs.max(dim=2)[0]

        # bg_socre = torch.sub(1, probs.max(dim=2)[0].max(dim=1, keepdim=True)[0])
        # cls_score = torch.cat([bg_socre, cls_score], dim=1)
        # cls_score = cls_score / cls_score.sum(dim=1, keepdim=True)
        if save_outs:
            return cls_score, distances, emb_vectors, reps
        return cls_score, distances

def build_emb_module(input_channels, emb_channels):
    emb_list = []
    emb_list.append(nn.BatchNorm1d(input_channels)),
    for i in range(len(emb_channels)):
        if i == 0:
            emb_list.append(nn.Linear(input_channels, emb_channels[i])),
            emb_list.append(nn.BatchNorm1d(emb_channels[i])),
            # self.emb.append(self.relu)
        else:
            emb_list.append(nn.Linear(emb_channels[i - 1], emb_channels[i])),
            if i != len(emb_channels) - 1:
                emb_list.append(nn.BatchNorm1d(emb_channels[i])),
                # self.emb.append(self.relu)

    for m in emb_list:
        if isinstance(m, nn.Linear):
            normal_init(m, std=0.01)
        elif isinstance(m, nn.BatchNorm2d):
            constant_init(m, 1)

    return nn.Sequential(*tuple(emb_list))

@HEADS.register_module
class DMLBBoxHead(nn.Module):
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
                 grad_scale=None,
                 alpha=0.3,
                 cls_emb_head_cfg=dict(
                     emb_channels=(256, 128),
                     num_modes=1,
                     sigma=0.5,
                     cls_norm=False,
                 ),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(DMLBBoxHead, self).__init__()
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

        self.cls_emb_head_cfg = cls_emb_head_cfg
        self.alpha = alpha

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        if self.with_cls:
            # emb_module = build_emb_module(in_channels, cls_emb_head_cfg['emb_channels'])
            self.cls_head = DMLHead(None, self.num_classes - 1, **cls_emb_head_cfg)

        if self.with_reg:
            base_ids = cls_emb_head_cfg['base_ids'] if 'base_ids' in cls_emb_head_cfg.keys() else None,
            novel_ids = cls_emb_head_cfg['novel_ids'] if 'novel_ids' in cls_emb_head_cfg.keys() else None,
            fc_cls_channels = (len(base_ids)+len(novel_ids)+1) if base_ids is not None else num_classes
            out_dim_reg = 4 if reg_class_agnostic else 4 * fc_cls_channels
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

        self.grad_scale = grad_scale

    def init_weights(self):
        # conv layers are already initialized by ConvModule
        # if self.with_cls:
        #     nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        #     nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.training and (self.grad_scale is not None):
            x = scale_tensor_gard(x, self.grad_scale)

        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score, distance = self.cls_head(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None

        if self.training:
            return cls_score, bbox_pred, distance
        else:
            return cls_score, bbox_pred

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

    @force_fp32(apply_to=('cls_score', 'bbox_pred', 'distance'))
    def loss(self,
             cls_score,
             bbox_pred,
             distance,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if cls_score is not None:
            if cls_score.numel() > 0:
                # losses['loss_cls'] = self.loss_cls(
                #     cls_score,
                #     labels,
                #     label_weights,
                #     avg_factor=avg_factor,
                #     reduction_override=reduction_override)
                losses['loss_cls'] = self.loss_dml_cls(cls_score, labels, label_weights, avg_factor)
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
        losses['emb'] = self.loss_emb(distance, labels, label_weights)
        return losses

    def loss_emb(self, distance, labels, label_weights):
        pos_num = (labels > 0).sum()
        pos_inds = labels > -1
        distance_pos = distance[pos_inds]

        n = distance_pos.size(0)
        n_cls = distance_pos.size(1)
        n_modes = distance_pos.size(2)

        labels_pos = labels[pos_inds]
        label_weights_pos = label_weights[pos_inds]
        labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=self.num_classes).byte()
        labels_pos_one_hot_inverse = torch.sub(1, labels_pos_one_hot)
        distance_pos_target = distance_pos[labels_pos_one_hot].reshape(n, -1, n_modes)
        distance_pos_other = distance_pos[labels_pos_one_hot_inverse].reshape(n, -1, n_modes)
        loss_emb_all = F.relu(distance_pos_target.min(-1)[0].squeeze(-1) - distance_pos_other.min(-1)[0].min(-1)[0] + self.alpha)
        loss_emb = (loss_emb_all*label_weights_pos) / pos_num
        return loss_emb

    def loss_dml_cls(self, cls_score, labels, label_weights, avg_factor):
        loss_cls_all = F.nll_loss(cls_score.log(), labels, None, None, -100, None, 'none') * label_weights
        loss_dml_cls = loss_cls_all.sum() / avg_factor
        return loss_dml_cls

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
        # scores = F.softmax(cls_score, dim=1) if cls_score is not None else None
        scores = cls_score

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
