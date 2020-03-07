import torch

from core.utils.metric_logger import Metric, MetricList


class InsAccuracy(Metric):
    """Instance accuracy"""
    name = 'ins_acc'

    def __init__(self, mode=None):
        super(InsAccuracy, self).__init__()
        self.mode = mode
        if mode is not None:
            assert mode in ['pos', 'neg']
            self.name = '{}_{}'.format(self.name, mode)

    def update_dict(self, preds, labels):
        ins_logit = preds['ins_logit']
        ins_label = labels['ins_label']
        valid_mask = labels['valid_mask']
        if ins_logit.size(1) == 1:
            pred_label = (ins_logit > 0.).squeeze().long()
        else:
            pred_label = ins_logit.argmax(1)
        tp_mat = pred_label.eq(ins_label)
        full_mat = ins_label.new_ones(ins_label.size()).byte()

        tp_mat *= (valid_mask > 0)
        full_mat *= (valid_mask > 0)

        if self.mode == 'pos':
            tp_mat = tp_mat * (ins_label > 0)
            full_mat = full_mat * (ins_label > 0)
        elif self.mode == 'neg':
            tp_mat = tp_mat * (ins_label == 0)
            full_mat = full_mat * (ins_label == 0)

        num_tp = tp_mat.sum().item()
        num_gt = full_mat.sum().item()
        self.update(num_tp, num_gt)


class Sample(Metric):
    """Sample"""
    name = 'sample'

    def __init__(self, mode=None):
        super(Sample, self).__init__()
        self.mode = mode
        if mode is not None:
            assert mode in ['pos', 'neg']
            self.name = '{}_{}'.format(self.name, mode)

    def update_dict(self, preds, labels):
        ins_label = labels['ins_label']
        valid_mask = labels['valid_mask']
        full_mat = ins_label.new_ones(ins_label.size()).byte() * (valid_mask > 0)
        if self.mode == 'pos':
            full_mat = full_mat * (ins_label > 0)
        elif self.mode == 'neg':
            full_mat = full_mat * (ins_label == 0)

        num_tp = full_mat.sum().item()
        num_gt = ins_label.size(0)
        self.update(num_tp, num_gt)


class Density(Metric):
    """Density"""
    name = 'density'

    def __init__(self):
        super(Density, self).__init__()

    def update_dict(self, preds, labels):
        # (batch_size, 3, num_centroids, num_neighbours)
        neighbour_xyz = labels['neighbour_xyz']
        # (batch_size, num_centroids, num_neighbours)
        neighbour_index = labels['neighbour_index']
        # (batch_size, 3, num_centroids)
        centroid_xyz = labels['centroid_xyz']

        num_inst = ((neighbour_xyz - centroid_xyz.unsqueeze(-1)).abs().sum(1) > 1e-5).sum().item() + 1
        num_sum = neighbour_index.numel()
        self.update(num_inst, num_sum)


class SegIoU(Metric):
    """Segmentation IoU
    References: https://github.com/pytorch/vision/blob/master/references/segmentation/utils.py
    """
    name = 'seg_IoU'

    def __init__(self, num_classes, ignore_index=-100):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.mat = None
        self.total = None

    def update_dict(self, preds, labels):
        # ins_logit, (batch_size, 2, num_centroids, num_neighbours)
        ins_logit = preds['ins_logit']
        # ins_label, (batch_size, num_centroids, num_neighbours)
        ins_label = labels['ins_label']
        # valid_mask, (batch_size, num_centroids, num_neighbours)
        valid_mask = labels['valid_mask']
        batch_size, num_centroids, num_neighbours = ins_label.size()
        pred_label = ins_logit.argmax(1)
        pred_label = pred_label.view(-1, num_neighbours)
        ins_label = ins_label.view(-1, num_neighbours)

        mask = valid_mask.byte().view(-1, num_neighbours)
        ins_label = ins_label[mask]
        pred_label = pred_label[mask]

        # Update confusion matrix
        # TODO: Compare the speed between torch.histogram and torch.bincount in pytorch v1.1.0
        n = self.num_classes
        with torch.no_grad():
            if self.mat is None:
                self.mat = ins_label.new_zeros((n, n))
            inds = n * ins_label + pred_label
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            if self.total is None:
                self.total = ins_label.new_zeros((1,)).float()
            self.total += batch_size * num_centroids

    def reset(self):
        self.mat = None
        self.total = None

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iou = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iou

    def __str__(self):
        acc_global, acc, iou = self.compute()
        if self.num_classes == 2:
            str_msg = 'all: {:.4f} pos: {:.4f} neg: {:.4f}'
            str_msg = str_msg.format(
                iou.mean().item(),
                iou.tolist()[1],
                iou.tolist()[0]
            )
        else:
            str_msg = 'mean_iou: {mean_iou:.4f} iou: {iou}'
            str_msg = str_msg.format(
                mean_iou=iou.mean().item(),
                iou=', '.join(['{:.4f}'.format(i) for i in iou.tolist()]),
            )
        return str_msg

    @property
    def summary_str(self):
        return str(self)

    @property
    def global_avg(self):
        acc_global, acc, iou = self.compute()
        avg_dict = dict(
            all=iou.mean().item(),
            pos=iou.tolist()[1],
            neg=iou.tolist()[0]
        )
        return avg_dict


class SegAcc(SegIoU):

    def __init__(self, num_classes, ignore_index=-100):
        super(SegAcc, self).__init__(num_classes, ignore_index)
        self.name = 'seg_acc'

    def __str__(self):
        acc_global, acc, iou = self.compute()
        if self.num_classes == 2:
            str_msg = 'all:{:.4f} in {:d}, pos:{:.4f} in {:d}, neg: {:.4f} in {:d}'
            str_msg = str_msg.format(
                acc_global.item(),
                (self.mat.float().sum()/self.total).int().item(),
                acc.tolist()[1],
                (self.mat.float().sum(1)/self.total).int().tolist()[1],
                acc.tolist()[0],
                (self.mat.float().sum(1)/self.total).int().tolist()[0],
            )
        else:
            str_msg = 'global_acc: {global_acc:.4f} acc: {acc} ' \
                      'total_num: {total_num:d} num: {num}'
            str_msg = str_msg.format(
                global_acc=acc_global.item(),
                acc=', '.join(['{:.4f}'.format(i) for i in acc.tolist()]),
                total_num=(self.mat.float().sum()/self.total).int().item(),
                num=', '.join(['{:d}'.format(i) for i in (self.mat.float().sum(1)/self.total).int().tolist()])
            )
        return str_msg

    @property
    def global_avg(self):
        acc_global, acc, iou = self.compute()
        avg_dict = dict(
            all=acc_global.item(),
            pos=acc.tolist()[1],
            neg=acc.tolist()[0]
        )
        return avg_dict


class InsComAccuracy(MetricList):

    def __init__(self):
        metrics = [SegIoU(2),
                   SegAcc(2),
                   InsAccuracy(), InsAccuracy('pos'), InsAccuracy('neg'),
                   Sample(), Sample('pos'), Sample('neg'),
                   Density()]
        super(InsComAccuracy, self).__init__(metrics)

