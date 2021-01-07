import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred, target, gamma, alpha)
    # TODO: find a proper way to handle the shape of weight
    if weight is not None:
        weight = weight.view(-1, 1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def no_sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    # self.alpha = self.alpha.to(preds.device)
    # alpha_ = torch.zeros(pred.size(-1)).cuda()
    # alpha_[0] += alpha
    # alpha_[1:] += (1 - alpha)
    # alpha_ = alpha_.to(pred.device)
    # # preds_softmax = F.softmax(pred, dim=1)
    # preds_softmax = pred
    # preds_logsoft = torch.log(preds_softmax)
    # preds_softmax = preds_softmax.gather(1, target.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
    # preds_logsoft = preds_logsoft.gather(1, target.view(-1, 1))
    # alpha_ = alpha_.gather(0, target.view(-1))
    # loss = -torch.mul(torch.pow((1 - preds_softmax), gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
    # loss = torch.mul(alpha_, loss.t())
    # print('before: ', loss)


    # ce_loss = F.cross_entropy(inputs, targets,
    #                           reduction='none', ignore_index=self.ignore_index)
    loss = F.nll_loss(pred.log(), target, None, None, -100, None, 'none')
    pt = torch.exp(-loss)
    focal_loss = alpha * torch.sub(1, pt)**gamma * loss

    if weight is not None:
        weight = weight.view(-1, 1)
    focal_loss = weight_reduce_loss(focal_loss, weight, reduction, avg_factor)

    # pred_sigmoid = pred.sigmoid()
    # pred_sigmoid = pred
    # target = target.type_as(pred)
    # pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # focal_weight = (alpha * target + (1 - alpha) *
    #                 (1 - target)) * pt.pow(gamma)
    # loss = F.nll_loss(pred.log(), target, None, None, -100, None, 'none') * focal_weight
    # # loss = F.binary_cross_entropy_with_logits(
    # #     pred, target, reduction='none') * focal_weight
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return focal_loss

@LOSSES.register_module
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            # raise NotImplementedError
            loss_cls = self.loss_weight * no_sigmoid_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        return loss_cls
