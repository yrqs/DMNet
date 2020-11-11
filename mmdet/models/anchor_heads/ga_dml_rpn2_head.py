import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init

from mmdet.ops import ConvModule, MaskedConv2d
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
from mmdet.core import (AnchorGenerator, anchor_inside_flags, anchor_target,
                        delta2bbox, force_fp32, ga_loc_target, ga_shape_target,
                        multi_apply, multiclass_nms)

from ..builder import build_loss
from ..losses import smooth_l1_loss

@HEADS.register_module
class GADMLRPN2Head(GuidedAnchorHead):
    """Guided-Anchor-based RetinaNet head."""
    def __init__(self,
                 num_classes,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=None,
                 emb_sizes=(2048, 1024),
                 num_modes=5,
                 sigma=0.5,
                 finetuning=False,
                 finetuning_num_classes=4,
                 loss_emb=dict(type='RepMetLoss', alpha=0.15, loss_weight=1.0),
                 **kwargs):
        # self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.emb_sizes = emb_sizes
        self.num_modes = num_modes
        self.sigma = sigma
        self.finetuning = finetuning
        self.finetuning_num_classes = finetuning_num_classes
        super(GADMLRPN2Head, self).__init__(num_classes, in_channels, **kwargs)
        self.loss_emb = build_loss(loss_emb)
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)

        self.rep = nn.Linear(1, (self.num_classes-1) * self.num_modes * self.emb_sizes[-1])
        self.representations = nn.Parameter(
            torch.FloatTensor(self.num_classes-1, self.num_modes, self.emb_sizes[-1]),
            requires_grad=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.emb = nn.ModuleList()
        for i in range(len(self.emb_sizes)):
            if i == 0:
                self.emb.append(nn.Conv2d(self.feat_channels, self.emb_sizes[i], 1, stride=1)),
                self.emb.append(nn.BatchNorm2d(self.emb_sizes[i])),
                self.emb.append(self.relu)
            else:
                self.emb.append(nn.Conv2d(self.emb_sizes[i-1], self.emb_sizes[i], 1, stride=1)),
                if i != len(self.emb_sizes) - 1:
                    self.emb.append(nn.BatchNorm2d(self.emb_sizes[i])),
                    self.emb.append(self.relu)

        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)
        self.conv_shape_cls = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)

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

        # self.finetuning_test = nn.Parameter(
        #     torch.zeros(1),
        #     requires_grad=True
        # )

    def init_weights(self):
        # for m in self.cls_convs:
        #     normal_init(m.conv, std=0.01)
        # for m in self.reg_convs:
        #     normal_init(m.conv, std=0.01)

        normal_init(self.rpn_conv, std=0.01)
        for m in self.emb:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        normal_init(self.rep, std=0.01)

        self.feature_adaption_cls.init_weights()
        self.feature_adaption_reg.init_weights()
        # self.feature_adaption_reg.init_weights()

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape_cls, std=0.01)
        normal_init(self.conv_shape_reg, std=0.01)
        # normal_init(self.dml_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

        # self.prob_factor.data.fill_(-15)

    def forward_single(self, x):
        # for name, param in self.named_parameters():
        #     print(name, ' : ', param.requires_grad)

        x = self.rpn_conv(x)

        loc_pred = self.conv_loc(x)
        shape_pred_cls = self.conv_shape_cls(x)
        shape_pred_reg = self.conv_shape_reg(x)

        feat_cls = self.feature_adaption_cls(x, shape_pred_cls)
        feat_reg = self.feature_adaption_reg(x, shape_pred_reg)

        if not self.training:
            # mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
            mask = loc_pred.sigmoid()[0] >= 0.1
            # mask = None
            # print(mask)
        else:
            mask = None
        # cls_score = self.retina_cls(cls_feat, mask)
        bbox_pred = self.retina_reg(feat_reg, mask)

        if self.training:
            reps = self.rep(torch.tensor(1.0).to(feat_reg.device).unsqueeze(0)).squeeze(0)
            reps  = reps.view((self.num_classes-1), self.num_modes, self.emb_sizes[-1])
            reps = F.normalize(reps, p=2, dim=2)
            if not self.finetuning:
                self.representations.data = reps.detach()
        else:
            reps = self.representations.detach()

            # reps = self.rep(torch.tensor(1.0).to(feat.device).unsqueeze(0)).squeeze(0)
            # reps  = reps.view((self.num_classes-1), self.num_modes, self.emb_sizes[-1])
            # reps = F.normalize(reps, p=2, dim=2)
        emb_vectors = feat_cls
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
        # probs_fg = probs_cls.max(2)[0]
        # print(probs_fg)
        # print(probs_fg.size())
        if not (mask is None):
        #     # print('a: ', probs_fg.size())
        #     # print('b: ', mask.size())
            probs_fg = probs_fg * mask.unsqueeze(0).float()
            # print(probs_fg)
        # cls_score = probs_fg
        # print(probs_fg)
        probs_bg = torch.sub(1, probs_cls.max(1)[0].max(1, keepdim=True)[0])
        # probs_bg = -(probs_fg.max(1, keepdim=True)[0] - 1)
        cls_score = torch.cat((probs_bg, probs_fg), 1)
        # print(cls_score.softmax(1))
        if self.training:
            if self.finetuning:
                return cls_score, bbox_pred, (shape_pred_cls, shape_pred_reg), loc_pred, emb_vectors
            else:
                return cls_score, bbox_pred, (shape_pred_cls, shape_pred_reg), loc_pred, distances
        else:
            # cls_score = cls_score.softmax(1)
            # print(reps)
            # print(distances[mask.unsqueeze(0).unsqueeze(0).expand_as(distances)])
            return cls_score.log(), bbox_pred, shape_pred_reg, loc_pred

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

        # print(type(shape_preds))
        shape_preds_cls = []
        shape_preds_reg = []
        for s in shape_preds:
            # print(type(s))
            shape_preds_cls.append(s[0])
            shape_preds_reg.append(s[1])

        # get squares and guided anchors
        # get sampled approxes
        approxs_list_cls, inside_flag_list_cls = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg, device=device)
        approxs_list_reg, inside_flag_list_reg = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg, device=device)
        squares_list_cls, guided_anchors_list_cls, _ = self.get_anchors(
            featmap_sizes, shape_preds_cls, loc_preds, img_metas, device=device)

        # get shape targets
        sampling = False if not hasattr(cfg, 'ga_sampler') else True
        shape_targets_cls = ga_shape_target(
            approxs_list_cls,
            inside_flag_list_cls,
            squares_list_cls,
            gt_bboxes,
            img_metas,
            self.approxs_per_octave,
            cfg,
            sampling=sampling)

        if shape_targets_cls is None:
            return None

        (bbox_anchors_list_cls, bbox_gts_list_cls, anchor_weights_list_cls, anchor_fg_num_cls,
         anchor_bg_num_cls) = shape_targets_cls

        anchor_total_num_cls = (
            anchor_fg_num_cls if not sampling else anchor_fg_num_cls + anchor_bg_num_cls)

        # get anchor targets
        sampling = False if self.cls_focal_loss else True
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else self.num_classes

        cls_reg_targets_cls = anchor_target(
            guided_anchors_list_cls,
            inside_flag_list_cls,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets_cls is None:
            return None
        (labels_list, label_weights_list, _, _,
         num_total_pos_cls, num_total_neg_cls) = cls_reg_targets_cls

        squares_list_reg, guided_anchors_list_reg, _ = self.get_anchors(
            featmap_sizes, shape_preds_reg, loc_preds, img_metas, device=device)
        shape_targets_reg = ga_shape_target(
            approxs_list_reg,
            inside_flag_list_reg,
            squares_list_reg,
            gt_bboxes,
            img_metas,
            self.approxs_per_octave,
            cfg,
            sampling=sampling)
        if shape_targets_reg is None:
            return None
        (bbox_anchors_list_reg, bbox_gts_list_reg, anchor_weights_list_reg, anchor_fg_num_reg,
         anchor_bg_num_reg) = shape_targets_reg
        anchor_total_num_reg = (
            anchor_fg_num_reg if not sampling else anchor_fg_num_reg + anchor_bg_num_reg)

        cls_reg_targets_reg = anchor_target(
            guided_anchors_list_reg,
            inside_flag_list_reg,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets_reg is None:
            return None
        (_, _, bbox_targets_list, bbox_weights_list,
         num_total_pos_reg, num_total_neg_reg) = cls_reg_targets_reg
        # print('a: ', [type(_) for _ in guided_anchors_list_cls])
        # print('b: ', [type(_) for _ in inside_flag_list])

        # num_total_samples = (
        #     num_total_pos if self.cls_focal_loss else num_total_pos +
        #     num_total_neg)
        num_total_samples_reg = num_total_pos_reg
        num_total_samples_cls = num_total_pos_cls

        # return dict(losses=self.finetuning_test)
        if self.finetuning:
            emb_vectors = extras
            emb_vectors_list_per_cls = [list() for _ in range(1, self.num_classes-1)]
            for i in range(len(emb_vectors)):
                emb_vectors_list = self.emb_vector_per_cls(
                    emb_vectors[i],
                    labels_list[i])
                for j in range(len(emb_vectors_list)):
                    emb_vectors_list_per_cls[j].append(emb_vectors_list[j])
            ev_flattens = []
            for ev in emb_vectors_list_per_cls:
                if len(ev) == 0:
                    ev_flattens.append(torch.tensor([]).to(device))
                else:
                    ev_flattens.append(torch.cat(ev, 0))
            for i, ev_f in enumerate(ev_flattens):
                if ev_f.size(0) != 0:
                    # print(ev_f)
                    new_r = ev_f.mean(0)
                    # new_r = (new_r + self.representations[i, 0, :]) / 2.0
                    # new_r = ev_f[0]
                    print(i)
                    self.representations[i, :, :] = new_r.expand(self.representations.size(1), -1)
            # return dict(losses=torch.tensor([0.0]).to(device))
            return dict(losses=self.finetuning_test)

        # print('2')
        # get classification and bbox regression losses
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=(num_total_samples_cls, num_total_samples_reg),
            cfg=cfg)

        # print('3')
        # print('num_total_samples: ', num_total_samples)
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

        # print('4')
        # get anchor shape loss
        losses_shape_cls = []
        for i in range(len(shape_preds_cls)):
            loss_shape_cls = self.loss_shape_single(
                shape_preds_cls[i],
                bbox_anchors_list_cls[i],
                bbox_gts_list_cls[i],
                anchor_weights_list_cls[i],
                anchor_total_num=anchor_total_num_cls)
            losses_shape_cls.append(loss_shape_cls)

        losses_shape_reg = []
        for i in range(len(shape_preds_reg)):
            loss_shape_reg = self.loss_shape_single(
                shape_preds_reg[i],
                bbox_anchors_list_reg[i],
                bbox_gts_list_reg[i],
                anchor_weights_list_reg[i],
                anchor_total_num=anchor_total_num_reg)
            losses_shape_reg.append(loss_shape_reg)

        # print('5')
        # get anchor embedding loss
        if not self.finetuning:
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
        else:
            losses_emb = torch.tensor([0.0]).to(device)

        if not self.finetuning:
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_shape_cls=losses_shape_cls,
                loss_shape_reg=losses_shape_reg,
                loss_loc=losses_loc,
                loss_emb=losses_emb)
        else:
            return dict(
                loss_cls=losses_cls,
                loss_bbox=losses_bbox,
                loss_shape_cls=losses_shape_cls,
                loss_shape_reg=losses_shape_reg,
                loss_loc=losses_loc)
            # return dict(
            #     loss_cls=[l*0.0 for l in losses_cls],
            #     loss_bbox=[l*0.0 for l in losses_bbox],
            #     loss_shape=[l*0.0 for l in losses_shape],
            #     loss_loc=[l*0.0 for l in losses_loc],
            #     loss_emb=[l*0.0 for l in losses_emb],)
            # return dict(
            #     loss_cls=[l * 0.0 for l in losses_cls],
            #     loss_bbox=[l * 0.0 for l in losses_bbox],
            #     loss_shape=[l * 0.0 for l in losses_shape],
            #     loss_loc=[l * 0.0 for l in losses_loc],
            #     loss_emb=[l * 0.0 for l in losses_emb], )

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        num_total_samples_cls, num_total_samples_reg = num_total_samples
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        # loss_cls_all = self.loss_cls(
        #     cls_score, labels, label_weights, avg_factor=num_total_samples)

        # loss_cls_all = F.cross_entropy(cls_score, labels, reduction='none') * label_weights
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
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples_cls

        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples_reg)
        return loss_cls[None], loss_bbox

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
            avg_factor=max(1, int(distance_pos.size(0))))
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