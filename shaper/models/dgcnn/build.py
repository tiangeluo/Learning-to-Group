from .dgcnn_cls import DGCNNCls
from .dgcnn_part_seg import DGCNNPartSeg
from ..loss import ClsLoss, SegLoss
from ..metric import ClsAccuracy, SegAccuracy


def build_dgcnn(cfg):
    if cfg.TASK == "classification":
        kwargs_dict = dict(cfg.MODEL.DGCNN)
        loss_kwargs = kwargs_dict.pop('loss')
        net = DGCNNCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            **kwargs_dict,
        )
        loss_fn = ClsLoss(loss_kwargs.label_smoothing)
        metric = ClsAccuracy()
    elif cfg.TASK == "part_segmentation":
        net = DGCNNPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            **cfg.MODEL.DGCNN,
        )
        loss_fn = SegLoss()
        metric = SegAccuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric
