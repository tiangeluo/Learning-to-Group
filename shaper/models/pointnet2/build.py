from .pn2_ssg_cls import PointNet2SSGCls
from .pn2_msg_cls import PointNet2MSGCls
from .pn2_ssg_part_seg import PointNet2SSGPartSeg
from .pn2_msg_part_seg import PointNet2MSGPartSeg
from ..loss import ClsLoss, SegLoss
from ..metric import ClsAccuracy, SegAccuracy


def build_pointnet2ssg(cfg):
    if cfg.TASK == 'classification':
        net = PointNet2SSGCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            **cfg.MODEL.PN2SSG,
        )
        loss_fn = ClsLoss()
        metric = ClsAccuracy()
    elif cfg.TASK == 'part_segmentation':
        net = PointNet2SSGPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            **cfg.MODEL.PN2SSG,
        )
        loss_fn = SegLoss()
        metric = SegAccuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric


def build_pointnet2msg(cfg):
    if cfg.TASK == 'classification':
        net = PointNet2MSGCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            **cfg.MODEL.PN2MSG,
        )
        loss_fn = ClsLoss()
        metric = ClsAccuracy()
    elif cfg.TASK == 'part_segmentation':
        net = PointNet2MSGPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            **cfg.MODEL.PN2MSG,
        )
        loss_fn = SegLoss()
        metric = SegAccuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric
