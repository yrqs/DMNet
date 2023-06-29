import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weighted_loss
import math


@weighted_loss
def triplet_loss(distance, label, alpha):
    # distance: (n, num_class, num_mode)
    # label: (n, 1)
    if label.size(0) == 0:
        loss = torch.tensor(0.0).to(label.device)
    else:
        label_ = label-1
        cls_gt_idx = torch.zeros(distance.size(0), distance.size(1)).to(label_.device).scatter(1, label_, 1).byte()
        cls_other_idx = torch.ones(distance.size(0), distance.size(1)).to(label_.device).scatter(1, label_, 0).byte()
        dis_cls_gt = distance[cls_gt_idx].view(distance.size(0), -1, distance.size(2))
        dis_cls_other = distance[cls_other_idx].view(distance.size(0), -1, distance.size(2))
        dis_cls_gt_min = dis_cls_gt.min(1)[0].min(1)[0]
        dis_cls_other_min = dis_cls_other.min(1)[0].min(1)[0]
        loss = F.relu(dis_cls_gt_min - dis_cls_other_min + alpha)
    return loss


@LOSSES.register_module
class TripletLoss(nn.Module):

    def __init__(self, alpha=0.3, reduction='mean', loss_weight=1.0):
        super(TripletLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                distance,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_repmet = self.loss_weight * triplet_loss(
            distance,
            label,
            weight,
            alpha=self.alpha,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        # loss_repmet = self.loss_weight * repmet_loss(
        #     distance,
        #     label,
        #     alpha=self.alpha,
        #     weight=weight)
        # print(loss_repmet)
        return loss_repmet
