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

    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x

@HEADS.register_module
class GARetinaDMLHead23(GuidedAnchorHead):
    """Guided-Anchor-based RetinaNet head."""
    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 stacked_convs=4,
                 emb_sizes=(2048, 1024),
                 num_modes=5,
                 sigma=0.5,
                 freeze=False,
                 save_outs=False,
                 loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.emb_sizes = emb_sizes
        self.num_modes = num_modes
        self.sigma = sigma
        super(GARetinaDMLHead23, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_emb = build_loss(loss_emb)
        self.save_outs = save_outs

        if freeze:
            for c in [self.cls_convs, self.reg_convs, self.conv_loc, self.conv_shape,
                      self.feature_adaption_cls, self.feature_adaption_reg]:
                for p in c.parameters():
                    p.requires_grad = False

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)

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

        self.retina_reg = MaskedConv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

    def init_weights(self):
        for m in self.emb:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        normal_init(self.rep, std=0.01)

        normal_init(self.cls_feat_enhance, std=0.01)

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

        cls_feat_ori = cls_feat

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        loc_pred = self.conv_loc(cls_feat)
        shape_pred = self.conv_shape(reg_feat)

        cls_feat_enhance_pre = self.cls_feat_enhance(cls_feat)
        # cls_feat_enhance_pre = torch.cat([cls_feat_enhance_pre, loc_pred.detach()], dim=1)

        feat_cls = self.feature_adaption_cls(cls_feat_ori, cls_feat_enhance_pre)
        feat_reg = self.feature_adaption_reg(reg_feat, shape_pred)

        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None

        bbox_pred = self.retina_reg(feat_reg, mask)

        if self.training:
            reps = self.rep(torch.tensor(1.0).to(feat_reg.device).unsqueeze(0)).squeeze(0)
            reps  = reps.view((self.num_classes-1), self.num_modes, self.emb_sizes[-1])
            reps = F.normalize(reps, p=2, dim=2)
            self.representations.data = reps.detach()
        else:
            reps = self.representations.detach()

        emb_vectors = feat_cls

        for e in self.emb:
            emb_vectors = e(emb_vectors)
        emb_vectors = F.normalize(emb_vectors, p=2, dim=1)

        distances = emb_vectors.permute(0, 2, 3, 1).unsqueeze(3).unsqueeze(4)
        distances = distances.expand(-1, -1, -1, self.num_classes-1, self.num_modes, -1)

        distances = torch.sqrt(((distances - reps)**2).sum(-1)).permute(0, 3, 4, 1, 2).contiguous()

        probs_cls = torch.exp(-distances**2/(2.0*self.sigma**2))
        probs_cls_sumj = probs_cls.sum(2)
        probs_cls_sumij = probs_cls_sumj.sum(1, keepdim=True)
        probs_fg = probs_cls_sumj / probs_cls_sumij

        if not (mask is None):
            probs_fg = probs_fg * mask.unsqueeze(0).float()

        probs_bg = torch.sub(1, probs_cls.max(1)[0].max(1, keepdim=True)[0])

        cls_score = torch.cat((probs_bg, probs_fg), 1)

        if self.training:
            return cls_score, bbox_pred, shape_pred, loc_pred, distances
        else:
            if self.save_outs:
                return cls_score.log(), bbox_pred, shape_pred, loc_pred, cls_feat, reg_feat, feat_cls, feat_reg, emb_vectors, cls_feat_enhance_pre
            else:
                return cls_score.log(), bbox_pred, shape_pred, loc_pred


    def forward(self, feats):
        if self.training:
            return multi_apply(self.forward_single, feats)
        else:
            if self.save_outs:
                cls_scores, bbox_preds, shape_preds_reg, loc_preds, cls_feat, reg_feat, cls_feat_adp, reg_feat_adp, emb_vectors, cls_feat_enhance_pres = multi_apply(self.forward_single, feats)
                res = dict()
                res['cls_feat'] = cls_feat
                res['reg_feat'] = reg_feat
                res['cls_feat_adp'] = cls_feat_adp
                res['reg_feat_adp'] = reg_feat_adp
                res['cls_loc'] = loc_preds
                res['cls_feat_enhance'] = cls_feat_enhance_pres
                res['emb_vectors'] = emb_vectors
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
        apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds', 'extras'))
    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             extras,
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

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_shape_cls=losses_shape,
            loss_loc=losses_loc,
            loss_emb=losses_emb)

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
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
        # print(num_total_samples)
        # loss_cls = self.loss_cls(
        #     cls_score,
        #     labels,
        #     label_weights,
        #     avg_factor=num_total_samples
        # )

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
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels