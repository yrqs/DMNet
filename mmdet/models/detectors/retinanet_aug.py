from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn.functional as F


@DETECTORS.register_module
class RetinaNetAug(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetAug, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)

    def extract_feat(self, img, img_aug=None):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        if img_aug is not None:
            x_aug = self.backbone(img_aug)
            if self.with_neck:
                x_aug = self.neck(x_aug)
            return x, x_aug
        else:
            return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        img_aug = F.interpolate(img, (img.shape[-2]//2, img.shape[-1]//2))
        x, x_aug = self.extract_feat(img, img_aug=img_aug)
        x_aug = (None,) + x_aug[:-1]
        outs = self.bbox_head(x, x_aug)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses