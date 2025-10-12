import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
From https://github.com/ZhuHaoranEIS/DN-TOD/blob/0b110c0d84329f7bf4c199c7065dab6ff98327fb/mmdet/models/losses/iou_loss.py
"""


def reduce_loss(loss, reduction):
    """
    From https://github.com/ZhuHaoranEIS/DN-TOD/blob/0b110c0d84329f7bf4c199c7065dab6ff98327fb/mmdet/models/losses/utils.py
    Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def dn_diou_loss(pred, targets, weight=None, hyper=0.2, eps=1e-7, reduction='mean'):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    anx = hyper / 2
    w = targets[:,2]-targets[:,0]
    h = targets[:,3]-targets[:,1]
    loss_banck = []
    for i in [-1, 0 ,1]:
        target = targets.clone()
        target[:,0] = target[:,0] - anx * w * i
        target[:,2] = target[:,2] + anx * w * i
        target[:,1] = target[:,1] - anx * h * i
        target[:,3] = target[:,3] + anx * h * i
        # overlap
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]

        # union
        ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = ap + ag - overlap + eps

        # IoU
        ious = overlap / union

        # enclose area
        enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]

        c2 = cw**2 + ch**2 + eps

        b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
        b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
        b2_x1, b2_y1 = target[:, 0], target[:, 1]
        b2_x2, b2_y2 = target[:, 2], target[:, 3]

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
        rho2 = left + right

        # DIoU
        dious = ious - rho2 / c2
        loss = 1 - dious
        loss_banck.append(loss.reshape(-1,1))
    loss_banck = torch.cat(loss_banck, dim=1)
    loss = torch.min(loss_banck, dim=1)[0]   
    return weight_reduce_loss(loss, weight, reduction=reduction)


def diou_loss(pred, target, eps=1e-7, weight=None, reduction='mean'):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return weight_reduce_loss(loss, weight, reduction=reduction)


    
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """
    From https://github.com/ZhuHaoranEIS/DN-TOD/blob/0b110c0d84329f7bf4c199c7065dab6ff98327fb/mmdet/models/losses/utils.py
    Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss