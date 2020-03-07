import functools

import torch
import torch.nn.functional as F

from IPython import embed

import torch.nn as nn
from torch.autograd import Variable
# ---------------------------------------------------------------------------- #
# Distance
# ---------------------------------------------------------------------------- #
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        #if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def bpdist(feature):
    """Compute pairwise distances of features.

    Args:
        feature (torch.Tensor): (batch_size, channels, num_inst)

    Returns:
        distance (torch.Tensor): (batch_size, num_inst, num_inst)

    Notes:
        This method returns square distances, and is optimized for lower memory and faster speed.
        Sqaure sum is more efficient than gather diagonal from inner product.

    """
    square_sum = torch.sum(feature ** 2, 1, keepdim=True)
    square_sum = square_sum.transpose(1, 2) + square_sum
    distance = torch.baddbmm(square_sum, feature.transpose(1, 2), feature, alpha=-2.0)
    return distance


def bpdist2(feature1, feature2):
    """Compute pairwise distances of features.

    Args:
        feature1 (torch.Tensor): (batch_size, channels, num_inst1)
        feature2 (torch.Tensor): (batch_size, channels, num_inst2)

    Returns:
        distance (torch.Tensor): (batch_size, num_inst1, num_inst2)

    Notes:
        This method returns square distances, and is optimized for lower memory and faster speed.

    """
    square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
    square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
    square_sum = square_sum1.transpose(1, 2) + square_sum2
    distance = torch.baddbmm(square_sum, feature1.transpose(1, 2), feature2, alpha=-2.0)
    return distance


def pdist2(feature1, feature2):
    """Compute pairwise distances of features.

    Args:
        feature1 (torch.Tensor): (num_inst1, channels)
        feature2 (torch.Tensor): (num_inst2, channels)

    Returns:
        distance (torch.Tensor): (num_inst1, num_inst2)

    Notes:
        This method returns square distances.

    """
    square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
    square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
    square_sum = square_sum1 + square_sum2.transpose(0, 1)
    distance = torch.addmm(square_sum, feature1, feature2.transpose(0, 1), alpha=-2.0)
    return distance


# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #

def encode_one_hot(target, num_classes):
    """Encode integer labels into one-hot vectors

    Args:
        target (torch.Tensor): (N,)
        num_classes (int): the number of classes

    Returns:
        torch.FloatTensor: (N, C)

    """
    one_hot = target.new_zeros(target.size(0), num_classes)
    one_hot = one_hot.scatter(1, target.unsqueeze(1), 1)
    return one_hot.float()


def smooth_cross_entropy(input, target, label_smoothing):
    """Cross entropy loss with label smoothing

    Args:
        input (torch.Tensor): (N, C)
        target (torch.Tensor): (N,)
        label_smoothing (float):

    Returns:
        loss (torch.Tensor): scalar

    """
    assert input.dim() == 2 and target.dim() == 1
    assert isinstance(label_smoothing, float)
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (label_smoothing / num_classes)
    log_prob = F.log_softmax(input, dim=1)
    loss = (- smooth_one_hot * log_prob).sum(1).mean()
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

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


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

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
    # otherwise average the loss by avg_factor
    else:
        if reduction != 'mean':
            raise ValueError(
                'avg_factor can only be used with reduction="mean"')
        loss = loss.sum() / avg_factor
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        #loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        if weight is not None:
            loss = loss_func(pred, target, reduction='none', **kwargs)
            loss = torch.sum(loss * weight)/torch.clamp(torch.sum(weight),0e-6)
        else:
            loss = loss_func(pred, target, reduction=reduction, **kwargs)
        return loss

    return wrapper


focal_loss = FocalLoss(gamma=2)
cross_entropy = weighted_loss(F.cross_entropy)
binary_cross_entropy_with_logit = weighted_loss(F.binary_cross_entropy_with_logits)
l2_loss = weighted_loss(F.mse_loss)

# ---------------------------------------------------------------------------- #
# Indexing
# ---------------------------------------------------------------------------- #


def batch_index_select(input, index, dim):
    """Batch index_select

    References: https://discuss.pytorch.org/t/batched-index-select/9115/7

    Args:
        input (torch.Tensor): (b, ...)
        index (torch.Tensor): (b, n)
        dim (int):

    """
    assert index.dim() == 2, 'Index should be 2-dim.'
    assert input.size(0) == index.size(0), 'Mismatched batch size: {} vs {}'.format(input.size(0), index.size(0))
    batch_size = index.size(0)
    num_select = index.size(1)
    views = [1 for _ in range(input.dim())]
    views[0] = batch_size
    views[dim] = num_select
    expand_shape = list(input.shape)
    expand_shape[dim] = -1
    index = index.view(views).expand(expand_shape)
    return torch.gather(input, dim, index)
