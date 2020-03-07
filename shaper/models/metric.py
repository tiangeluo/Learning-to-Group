from core.utils.metric_logger import Metric


class ClsAccuracy(Metric):
    """Classification accuracy"""
    name = 'cls_acc'

    def update_dict(self, preds, labels):
        cls_logit = preds['cls_logit']  # (batch_size, num_classes)
        cls_label = labels['cls_label']  # (batch_size,)
        pred_label = cls_logit.argmax(1)
        num_tp = pred_label.eq(cls_label).sum().item()
        num_gt = cls_label.numel()
        self.update(num_tp, num_gt)


class SegAccuracy(Metric):
    """Segmentation accuracy"""
    name = 'seg_acc'

    def __init__(self):
        super(SegAccuracy, self).__init__()

    def update_dict(self, preds, labels):
        seg_logit = preds['seg_logit']  # (batch_size, num_classes, num_points)
        seg_label = labels['seg_label']  # (batch_size, num_points)
        pred_label = seg_logit.argmax(1)
        tp_mask = pred_label.eq(seg_label)  # (batch_size, num_points)
        self.update(tp_mask.sum().item(), tp_mask.numel())
