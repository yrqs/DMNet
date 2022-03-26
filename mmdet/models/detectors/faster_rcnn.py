from ..registry import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

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
                 freeze_backbone=False,
                 freeze_rpn=False,
                 freeze_shared_head=False):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if freeze_rpn:
            for p in self.rpn_head.parameters():
                p.requires_grad = False

        if freeze_shared_head:
            for p in self.shared_head.parameters():
                p.requires_grad = False