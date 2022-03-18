import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init

from mmdet.ops import ConvModule, MaskedConv2d, DeformConv
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
from mmdet.core import (AnchorGenerator, anchor_inside_flags, anchor_target,
                        delta2bbox, force_fp32, ga_loc_target, ga_shape_target,
                        multi_apply, multiclass_nms)

from ..builder import build_loss
from ..losses import FocalLoss
import os

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class DMLNegHead(nn.Module):
    def __init__(self,
                 emb_module,
                 output_channels,
                 emb_channels,
                 num_modes,
                 sigma,
                 cls_norm,
                 beta=0.3,
                 neg_num_modes=2,
                 base_ids=None,
                 novel_ids=None,
                 freeze=False):
        assert num_modes==1
        super().__init__()
        self.num_reps = len(base_ids) + len(novel_ids) if base_ids is not None else output_channels
        self.output_channels = output_channels
        self.num_modes = num_modes
        self.sigma = sigma
        self.emb_module = emb_module
        self.emb_channels = emb_channels
        self.rep_fc = nn.Linear(1, self.num_reps * num_modes * emb_channels[-1])
        self.cls_norm = cls_norm
        self.beta = beta
        self.neg_num_modes = neg_num_modes
        self.representations = nn.Parameter(
            torch.FloatTensor(self.num_reps, self.num_modes, self.emb_channels[-1]),
            requires_grad=False)
        self.neg_offset_fc = nn.Linear(self.emb_channels[-1], self.emb_channels[-1] * neg_num_modes)
        self.base_ids=base_ids
        self.novel_ids=novel_ids

        normal_init(self.rep_fc, std=0.01)
        normal_init(self.neg_offset_fc, std=0.01)

        if freeze:
            for c in [self.neg_offset_fc]:
                for p in c.parameters():
                    p.requires_grad = False

    def forward(self, x, save_outs=False):
        emb_vectors = self.emb_module(x)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        if self.training:
            reps = self.rep_fc(torch.tensor(1.0).to(x.device).unsqueeze(0)).squeeze(0)
            reps  = reps.view(self.num_reps, self.num_modes, self.emb_channels[-1])
            reps = F.normalize(reps, p=2, dim=2)
            self.representations.data = reps.detach()
        else:
            reps = self.representations.detach()

        neg_offset = self.neg_offset_fc(reps.squeeze(1).abs()).view(reps.size(0), self.neg_num_modes, reps.size(-1))

        if self.base_ids is not None:
            reps = reps[self.base_ids, :, :]
            neg_offset = neg_offset[self.base_ids, :, :]

        reps_neg = neg_offset + reps.abs().expand_as(neg_offset)
        reps_neg = reps_neg * reps.sign().expand_as(neg_offset)
        reps_neg = F.normalize(reps_neg, p=2, dim=2)

        if self.training and self.output_channels > 20:
            reps_list = reps.split(20, dim=0)
            distance_list = []
            for r in reps_list:
                dis = self.calculate_distance(r, emb_vectors)
                distance_list.append(dis)
            distance = torch.cat(distance_list, dim=1)

            reps_neg_list = reps_neg.split(20, dim=0)
            distance_neg_list = []
            for rn in reps_neg_list:
                dis = self.calculate_distance(rn, emb_vectors)
                distance_neg_list.append(dis)
            distance_neg = torch.cat(distance_neg_list, dim=1)
        else:
            distance = self.calculate_distance(reps, emb_vectors)
            distance_neg = self.calculate_distance(reps_neg, emb_vectors)

        probs_neg = torch.exp(-distance_neg**2/(2.0*self.sigma**2))
        cls_score_neg = probs_neg.max(dim=2)[0]

        probs_ori = torch.exp(-(distance)**2/(2.0*self.sigma**2))
        probs_ori = probs_ori.max(dim=2)[0]
        probs = torch.exp(-(distance+self.beta*(2-distance_neg.min(dim=2, keepdim=True)[0]))**2/(2.0*self.sigma**2))
        if self.cls_norm:
            probs_sumj = probs.sum(2)
            probs_sumij = probs_sumj.sum(1, keepdim=True)
            cls_score = probs_sumj / probs_sumij
        else:
            cls_score = probs.max(dim=2)[0]

        if save_outs:
            return cls_score, cls_score_neg, distance, distance_neg, probs_ori, emb_vectors, reps, reps_neg
        return cls_score, cls_score_neg, distance, distance_neg, probs_ori

    def calculate_distance(self, reps, emb_vectors):
        reps_ex = reps[None, :, :, :, None, None].expand(emb_vectors.size(0), -1, -1, -1, emb_vectors.size(2), emb_vectors.size(3))
        emb_vectors_ex = emb_vectors[:, None, None, :, :, :].expand_as(reps_ex)
        distance = torch.sqrt(((emb_vectors_ex - reps_ex)**2).sum(3))
        return distance


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
class GARetinaDMLNegHead4(GuidedAnchorHead):
    """Guided-Anchor-based RetinaNet head."""
    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 stacked_convs=4,
                 freeze=False,
                 save_outs=False,
                 neg_sample_thresh=0.2,
                 pos_sub_neg_thresh=0.1,
                 alpha_hn=0.15,
                 neg_hn_ratio=1,
                 cls_emb_head_cfg=dict(emb_channels=(256, 128),
                                       num_modes=1,
                                       sigma=0.5,
                                       cls_norm=False,
                                       beta=0.3,
                                       neg_num_modes=2),
                 loss_emb=dict(type='TripletLoss', alpha=0.15, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cls_emb_head_cfg = cls_emb_head_cfg
        self.neg_sample_thresh = neg_sample_thresh
        self.pos_sub_neg_thresh = pos_sub_neg_thresh
        self.alpha_hn = alpha_hn
        self.neg_hn_ratio = neg_hn_ratio
        super().__init__(num_classes, in_channels, **kwargs)
        self.loss_emb = build_loss(loss_emb)
        self.save_outs = save_outs

        # for c in [self.cls_convs, self.reg_convs, self.conv_loc, self.conv_shape, self.cls_feat_enhance,
        #           self.feature_adaption_cls, self.feature_adaption_reg, self.cls_head]:
        #     for p in c.parameters():
        #         p.requires_grad = Falses

        if freeze:
            for c in [self.cls_convs, self.reg_convs, self.conv_loc, self.conv_shape,
                      self.feature_adaption_cls, self.feature_adaption_reg, self.cls_head]:
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

        self.feature_adaption_cls = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)

        self.feature_adaption_reg = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)

        self.retina_reg = MaskedConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        emb_module = build_emb_module(self.feat_channels, self.cls_emb_head_cfg['emb_channels'],
                                      kernel_size=3, padding=1, stride=1)
        self.cls_head = DMLNegHead(emb_module, self.num_classes-1, **self.cls_emb_head_cfg)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        self.feature_adaption_cls.init_weights()
        self.feature_adaption_reg.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        # for name, param in self.named_parameters():
        #     print(name, ' : ', param.requires_grad)
        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        if self.save_outs:
            feat_cls, offset_cls = self.feature_adaption_cls(cls_feat, shape_pred, self.save_outs)
            feat_reg, offset_reg = self.feature_adaption_reg(reg_feat, shape_pred, self.save_outs)
        else:
            feat_cls = self.feature_adaption_cls(cls_feat, shape_pred, self.save_outs)
            feat_reg = self.feature_adaption_reg(reg_feat, shape_pred, self.save_outs)

        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None

        bbox_pred = self.retina_reg(feat_reg, mask)

        cls_score, cls_score_neg, distance, distance_neg, probs_ori = self.cls_head(feat_cls)

        if self.training:
            return cls_score, bbox_pred, shape_pred, loc_pred, distance, cls_score_neg, distance_neg, probs_ori
        else:
            if self.save_outs:
                return cls_score, bbox_pred, shape_pred, loc_pred, cls_feat, reg_feat, feat_cls, feat_reg, offset_cls, offset_reg, cls_score
            else:
                return cls_score, bbox_pred, shape_pred, loc_pred


    def forward(self, feats):
        if self.training:
            return multi_apply(self.forward_single, feats)
        else:
            if self.save_outs:
                cls_scores, bbox_preds, shape_preds_reg, loc_preds, cls_feat, reg_feat, cls_feat_adp, reg_feat_adp, emb_vectors, cls_feat_enhance_pres, offsets_cls, offsets_reg, distances, probs_cls, cls_score = multi_apply(self.forward_single, feats)
                res = dict()
                res['cls_feat'] = cls_feat
                res['reg_feat'] = reg_feat
                res['cls_feat_adp'] = cls_feat_adp
                res['reg_feat_adp'] = reg_feat_adp
                res['cls_loc'] = loc_preds
                res['cls_feat_enhance'] = cls_feat_enhance_pres
                res['emb_vectors'] = emb_vectors
                res['offsets_cls'] = offsets_cls
                res['offsets_reg'] = offsets_reg
                res['distances'] = distances
                res['probs_cls'] = probs_cls
                res['cls_score'] = cls_score
                save_idx = 1
                save_path_base = 'mytest/ga_retina_dml3_feature.pth'
                save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
                while os.path.exists(save_path):
                    save_idx += 1
                    save_path = save_path_base[:-4] + str(save_idx) + save_path_base[-4:]
                torch.save(res, save_path)
            else:
                cls_scores, bbox_preds, shape_preds_reg, loc_preds = multi_apply(self.forward_single, feats)
            return cls_scores, bbox_preds, shape_preds_reg, loc_preds

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds', 'extras', 'cls_scores_neg', 'distances_neg', 'probs_ori'))
    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             extras,
             cls_scores_neg,
             distances_neg,
             probs_ori,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.approx_generators)

        device = cls_scores[0].device

        # get loc targets
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.octave_base_scale,
            self.anchor_strides,
            center_ratio=cfg.center_ratio,
            ignore_ratio=cfg.ignore_ratio)

        # get sampled approxes
        approxs_list, inside_flag_list = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg, device=device)
        # get squares and guided anchors
        squares_list, guided_anchors_list, _ = self.get_anchors(
            featmap_sizes, shape_preds, loc_preds, img_metas, device=device)

        # get shape targets
        sampling = False if not hasattr(cfg, 'ga_sampler') else True
        shape_targets = ga_shape_target(
            approxs_list,
            inside_flag_list,
            squares_list,
            gt_bboxes,
            img_metas,
            self.approxs_per_octave,
            cfg,
            sampling=sampling)
        if shape_targets is None:
            return None
        (bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num,
         anchor_bg_num) = shape_targets
        anchor_total_num = (
            anchor_fg_num if not sampling else anchor_fg_num + anchor_bg_num)

        # get anchor targets
        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            guided_anchors_list,
            inside_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos if self.cls_focal_loss else num_total_pos +
                                                      num_total_neg)

        # get classification and bbox regression losses
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
        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],
                loc_targets[i],
                loc_weights[i],
                loc_avg_factor=loc_avg_factor,
                cfg=cfg)
            losses_loc.append(loss_loc)

        # get anchor shape loss
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(
                shape_preds[i],
                bbox_anchors_list[i],
                bbox_gts_list[i],
                anchor_weights_list[i],
                anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)
        # print('5')
        # get anchor embedding loss
        distances = extras
        losses_emb = []
        num_total_samples_emb = sum([int((_>0).sum()) for _ in labels_list])
        for i in range(len(distances)):
            loss_emb = self.loss_emb_single(
                distances[i],
                labels_list[i],
                label_weights_list[i],
                num_total_samples=num_total_samples_emb)
            losses_emb.append(loss_emb)

        losses_pos_hn = []
        losses_pos_hnp = []
        losses_neg_hn = []
        num_pos_hn_list = []
        num_neg_hn_list = []
        for i in range(len(cls_scores)):
            loss_pos_hn, loss_pos_hnp, num_pos_hn, loss_neg_hn, num_neg_hn = self.loss_emb_neg_single(
                distances[i],
                distances_neg[i],
                labels_list[i],
                probs_ori[i],
                cls_scores[i],
            )
            losses_pos_hn.append(loss_pos_hn)
            losses_pos_hnp.append(loss_pos_hnp)
            num_pos_hn_list.append(num_pos_hn)
            losses_neg_hn.append(loss_neg_hn)
            num_neg_hn_list.append(num_neg_hn)

        num_pos_hn_all = max(1, sum(num_pos_hn_list))
        num_neg_hn_all = max(1, sum(num_neg_hn_list))

        losses_pos_hn = [_ / num_pos_hn_all for _ in losses_pos_hn]
        losses_pos_hnp = [_ / max(num_total_samples_emb, 1) for _ in losses_pos_hnp]
        losses_neg_hn = [_ / num_neg_hn_all for _ in losses_neg_hn]

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_shape_cls=losses_shape,
            loss_loc=losses_loc,
            loss_emb=losses_emb,
            loss_pos_hn=losses_pos_hn,
            loss_pos_hnp=losses_pos_hnp,
            loss_neg_hn=losses_neg_hn,
        )

    def loss_emb_neg_single(self, distance, neg_distance, labels, prob_ori, cls_score):
        distance = distance.flatten(3).permute(0, 3, 1, 2).flatten(0, 1)
        neg_distance = neg_distance.flatten(3).permute(0, 3, 1, 2).min(dim=-1, keepdim=True)[0].flatten(0, 1)

        distance_cat = torch.cat([distance, neg_distance], dim=-1)

        prob_ori = prob_ori.clone().detach()
        prob_ori = prob_ori.permute(0, 2, 3, 1).flatten(0, 2)
        cls_score = cls_score.clone().detach()
        cls_score = cls_score.permute(0, 2, 3, 1).flatten(0, 2)
        labels = labels.flatten(0)
        n = labels.size(0)

        labels_oh = F.one_hot(labels, num_classes=self.num_classes).byte()[:, 1:]

        pos_inds = labels > 0

        # loss_pos_hn
        if pos_inds.any():
            prob_ori_pos = prob_ori[pos_inds]
            cls_score_pos = cls_score[pos_inds]
            distance_cat_pos = distance_cat[pos_inds]
            labels_oh_pos = labels_oh[pos_inds]

            prob_ori_pos_other = prob_ori_pos.clone()
            prob_ori_pos_other[labels_oh_pos] = -1.
            prob_ori_pos_other_max_value, prob_ori_pos_other_max_ind = prob_ori_pos_other.max(dim=-1)
            pos_hn_inds = prob_ori_pos_other_max_value > self.neg_sample_thresh

            if pos_hn_inds.any():
                distance_cat_pos_hn = distance_cat_pos[pos_hn_inds]
                pos_hn_labels = prob_ori_pos_other_max_ind[pos_hn_inds]
                hn_labels_oh = F.one_hot(pos_hn_labels, num_classes=self.num_classes-1).byte()
                distance_cat_pos_hn = distance_cat_pos_hn[hn_labels_oh]
                loss_pos_hn = F.relu(distance_cat_pos_hn[:, 1] - distance_cat_pos_hn[:, 0] + self.alpha_hn)
                loss_pos_hn = loss_pos_hn.sum()
                num_pos_hn = pos_hn_inds.sum()
            else:
                loss_pos_hn = torch.zeros(1).to(distance.device)
                num_pos_hn = 0

            distance_cat_pos_hnp = distance_cat_pos[labels_oh_pos]
            loss_pos_hnp = F.relu(distance_cat_pos_hnp[:, 0] - distance_cat_pos_hnp[:, 1] + self.loss_emb.alpha)

        else:
            loss_pos_hnp = torch.zeros(1).to(distance.device)
            loss_pos_hn = torch.zeros(1).to(distance.device)
            num_pos_hn = 0

        neg_inds = (labels == 0)
        # loss_neg_hn
        if neg_inds.any():
            prob_ori_neg = prob_ori[neg_inds]
            distance_cat_neg = distance_cat[neg_inds]
            prob_ori_neg_max_value, prob_ori_neg_max_ind = prob_ori_neg.max(dim=-1)
            neg_hn_inds = prob_ori_neg_max_value > self.neg_sample_thresh
            if neg_hn_inds.any():
                distance_cat_neg_hn = distance_cat_neg[neg_hn_inds]
                neg_hn_labels = prob_ori_neg_max_ind[neg_hn_inds]
                hn_labels_oh = F.one_hot(neg_hn_labels, num_classes=self.num_classes-1).byte()
                distance_cat_neg_hn = distance_cat_neg_hn[hn_labels_oh]
                loss_neg_hn = F.relu(distance_cat_neg_hn[:, 1] - distance_cat_neg_hn[:, 0] + self.alpha_hn)
                num_neg_hn = min(loss_neg_hn.size(0), self.neg_hn_ratio * pos_inds.sum())
                loss_neg_hn = loss_neg_hn.sort(dim=0, descending=True)[0][:num_neg_hn]
                loss_neg_hn = loss_neg_hn.sum()
            else:
                loss_neg_hn = torch.zeros(1).to(distance.device)
                num_neg_hn = 0
        else:
            loss_neg_hn = torch.zeros(1).to(distance.device)
            num_neg_hn = 0

        return loss_pos_hn, loss_pos_hnp, num_pos_hn, loss_neg_hn, num_neg_hn


    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = inverse_sigmoid(cls_score)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
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

    def emb_vector_per_cls(self, emb_vector, label):
        if label.dim() == 1:
            label = label.unsqueeze(0)
        emb_vector = emb_vector.view(emb_vector.size(0), emb_vector.size(1), -1).permute(0, 2, 1)
        emb_vector_per_cls = []
        for i in range(self.finetuning_num_classes):
            emb_vector_per_cls.append(emb_vector[label==i+1])
        return emb_vector_per_cls

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
                scores = cls_score
                # scores = cls_score.sigmoid()
            else:
                # scores = cls_score.softmax(-1)
                scores = cls_score
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