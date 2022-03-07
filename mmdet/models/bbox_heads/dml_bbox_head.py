import torch.nn as nn

from mmdet.ops import ConvModule
from ..registry import HEADS
from .bbox_head import BBoxHead

import torch
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init
from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..losses import accuracy

class DMLHead(nn.Module):
    def __init__(self,
                 emb_module,
                 cls_num,
                 emb_channels,
                 num_modes,
                 sigma,
                 cls_norm,
                 freeze=False):
        assert num_modes==1
        super().__init__()
        self.cls_num = cls_num
        self.num_modes = num_modes
        self.sigma = sigma
        self.emb_module = emb_module
        self.emb_channels = emb_channels
        # self.rep_fc = nn.Linear(1, cls_num * num_modes * emb_channels[-1])
        self.cls_norm = cls_norm
        reps = nn.Embedding(cls_num * num_modes,  emb_channels[-1])
        normal_init(reps)
        self.representatives = reps.weight
        # normal_init(self.rep_fc, std=0.01)
        # constant_init(self.neg_offset_fc, 0)

        if freeze:
            for c in [self.neg_offset_fc]:
                for p in c.parameters():
                    p.requires_grad = False

    def forward(self, x, save_outs=False):
        emb_vectors = self.emb_module(x)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        reps = self.representatives.view(self.cls_num, self.num_modes, self.emb_channels[-1])
        reps = F.normalize(reps, p=2, dim=2)

        reps_ex = reps[None, :, :, :]
        emb_vectors_ex = emb_vectors[:, None, None, :]

        distances = torch.sqrt(((emb_vectors_ex - reps_ex)**2).sum(-1))

        # probs_ori = torch.exp(-(distances)**2/(2.0*self.sigma**2))
        # probs_ori = probs_ori.max(dim=2)[0]
        probs = torch.exp(-(distances)**2/(2.0*self.sigma**2))
        if self.cls_norm:
            probs_sumj = probs.sum(2)
            probs_sumij = probs_sumj.sum(1, keepdim=True)
            cls_score = probs_sumj / probs_sumij
        else:
            cls_score = probs.max(dim=2)[0]

        bg_socre = torch.sub(1, probs.max(2)[0].max(dim=1, keepdim=True)[0])
        cls_score = torch.cat([bg_socre, cls_score], dim=1)
        if save_outs:
            return cls_score, distances, emb_vectors, reps
        return cls_score, distances

def build_emb_module(input_channels, emb_channels):
    emb_list = []
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
class DMLFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 alpha=0.15,
                 cls_emb_head_cfg=dict(
                     emb_channels=(256, 128),
                     num_modes=1,
                     sigma=0.5,
                     cls_norm=False,
                 ),
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.alpha = alpha
        self.cls_emb_head_cfg = cls_emb_head_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(1, 1)
            for c in [self.fc_cls]:
                for p in c.parameters():
                    p.requires_grad = False

        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        if num_cls_fcs > 0 or num_shared_fcs > 0:
            emb_in_channels = self.fc_out_channels
        elif num_cls_convs > 0 or num_shared_convs > 0:
            emb_in_channels = self.conv_out_channels if self.with_avg_pool else self.conv_out_channels * self.roi_feat_area
        else:
            emb_in_channels = self.in_channels if self.with_avg_pool else self.in_channels * self.roi_feat_area

        emb_module = build_emb_module(emb_in_channels, self.cls_emb_head_cfg['emb_channels'])
        self.cls_head = DMLHead(emb_module, self.num_classes-1, **self.cls_emb_head_cfg)

    def init_weights(self):
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super().init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score, distance = self.cls_head(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        if self.training:
            return cls_score, bbox_pred, distance
        else:
            return cls_score, bbox_pred

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
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
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
        pos_inds = labels > 0
        distance_pos = distance[pos_inds]

        n = distance_pos.size(0)
        n_cls = distance_pos.size(1)
        n_modes = distance_pos.size(2)

        labels_pos = labels[pos_inds]
        label_weights_pos = label_weights[pos_inds]
        labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=self.num_classes).byte()[:, 1:]
        labels_pos_one_hot_inverse = torch.sub(1, labels_pos_one_hot)
        distance_pos_target = distance_pos[labels_pos_one_hot].reshape(n, -1, n_modes)
        distance_pos_other = distance_pos[labels_pos_one_hot_inverse].reshape(n, -1, n_modes)
        loss_emb_all = F.relu(distance_pos_target.min(-1)[0].squeeze(-1) - distance_pos_other.min(-1)[0].min(-1)[0] + self.alpha)
        loss_emb = (loss_emb_all*label_weights_pos).mean()
        return loss_emb

    def loss_dml_cls(self, cls_score, labels, label_weights, avg_factor):
        loss_cls_all = F.nll_loss(cls_score.log(), labels, None, None, -100, None, 'none') * label_weights
        loss_dml_cls = loss_cls_all.sum() / avg_factor
        return loss_dml_cls

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
        # scores = cls_score / cls_score.sum(1, keepdim=True) if cls_score is not None else None
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