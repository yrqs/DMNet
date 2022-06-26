from ..registry import DETECTORS
from .two_stage import TwoStageDetector
import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler, bbox2result
import torch.nn as nn
from mmdet.ops.roi_pool.roi_pool import RoIPool

@DETECTORS.register_module
class MetaFasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 multi_rois=False,
                 roi_offsets=None,
                 freeze_backbone=False,
                 freeze_rpn=False,
                 freeze_shared_head=False):
        super(MetaFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.multi_rois = multi_rois
        self.roi_offsets = roi_offsets

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_rpn:
            for p in self.rpn_head.parameters():
                p.requires_grad = False

        if freeze_shared_head:
            for p in self.shared_head.parameters():
                p.requires_grad = False
        self.roi_pooler = RoIPool((14, 14), 1.)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        assert len(img_metas) == 1

        x = self.extract_feat(img)

        have_support_data = 'support_instance_list' in img_metas[0].keys()
        if have_support_data:
            support_instance_list = img_metas[0]['support_instance_list']
            support_instance_labels = img_metas[0]['support_instance_labels'].to(img.device)
            support_instance_feats = []
            for si in support_instance_list:
                support_instance_feats.append(self.extract_feat(si['img'][None, ...].to(img.device))[0])
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_metas,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_metas, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
                if have_support_data:
                    instance_pooling_feats = []
                    for sif in support_instance_feats:
                        roi = torch.tensor([[0, 0., 0., sif.shape[3], sif.shape[2]]]).to(sif.device)
                        instance_pooling_feats.append(self.roi_pooler(sif, roi))
                    instance_pooling_feats = torch.cat(instance_pooling_feats, dim=0)
                    instance_propotypes = self.shared_head(instance_pooling_feats)

            if self.multi_rois:
                assert self.with_shared_head and (self.roi_offsets is not None)
                assert isinstance(self.roi_offsets, (tuple, list))
                gt_rois = bbox2roi(gt_bboxes)
                gt_rois_labels = torch.cat(gt_labels, dim=0)
                gt_rois_h = gt_rois[:, 3] - gt_rois[:, 1]
                gt_rois_w = gt_rois[:, 4] - gt_rois[:, 2]
                multi_rois = gt_rois[:, None, :].expand(-1, len(self.roi_offsets), -1)
                roi_offsets = torch.tensor(self.roi_offsets).to(gt_rois.device)[None, :, :].expand(multi_rois.size(0), -1, -1)
                w_h_w_h = torch.cat([gt_rois_w[:, None], gt_rois_h[:, None], gt_rois_w[:, None], gt_rois_h[:, None]], dim=1)
                roi_offsets = roi_offsets * w_h_w_h[:, None, :].expand_as(roi_offsets)
                multi_rois[:, :, 1:] = multi_rois[:, :, 1:] + roi_offsets
                multi_rois_feats = self.bbox_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], multi_rois.reshape(-1, 5))
                multi_rois_feats = self.shared_head(multi_rois_feats)
            # cls_score, bbox_pred = self.bbox_head(bbox_feats)
            if self.multi_rois:
                outs = self.bbox_head(bbox_feats, (multi_rois_feats, gt_rois_labels))
            elif have_support_data:
                outs = self.bbox_head(bbox_feats, (instance_propotypes, support_instance_labels))
            else:
                outs = self.bbox_head(bbox_feats)
            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_inputs = outs + bbox_targets
            # loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
            #                                 *bbox_targets)
            loss_bbox = self.bbox_head.loss(*loss_inputs)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]

            if mask_feats.shape[0] > 0:
                mask_pred = self.mask_head(mask_feats)
                mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                                pos_labels)
                losses.update(loss_mask)

        return losses

    def simple_test(self,
                    img,
                    img_metas,
                    proposals=None,
                    rescale=False,
                    pre_test=False,
                    support_instance_list=None,
                    support_instance_labels=None):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if pre_test:
            support_instance_labels = support_instance_labels.to(img.device)
            support_instance_feats = []
            for si in support_instance_list:
                support_instance_feats.append(self.extract_feat(si.to(img.device))[0])
            if self.with_shared_head:
                instance_pooling_feats = []
                for sif in support_instance_feats:
                    roi = torch.tensor([[0, 0., 0., sif.shape[3], sif.shape[2]]]).to(sif.device)
                    instance_pooling_feats.append(self.roi_pooler(sif, roi))
                instance_pooling_feats = torch.cat(instance_pooling_feats, dim=0)
                instance_pooling_feats = self.shared_head(instance_pooling_feats)
            self.bbox_head(None, (instance_pooling_feats, support_instance_labels), True)
            return

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_metas,
                                                 self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def finetune_init(self, x):
        x_c3 = self.backbone(x)
        rois = torch.tensor([[0., 0, 0, x.size(2) - 1, x.size(3) - 1]]).to(x.device)
        x_pool = self.bbox_roi_extractor(x_c3, rois.float())
        x_c5 = self.shared_head(x_pool)
        x_vector = self.bbox_head.finetune_init(x_c5)
        return x_vector