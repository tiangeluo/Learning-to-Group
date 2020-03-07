from .pointnet_cls import PointNetCls, PointNetClsLoss
from .pointnet_part_seg import PointNetPartSeg, PointNetPartSegLoss
from ..metric import ClsAccuracy, SegAccuracy


def build_pointnet(cfg):
    if cfg.TASK == 'classification':
        net = PointNetCls(
            in_channels=cfg.INPUT.IN_CHANNELS,
            out_channels=cfg.DATASET.NUM_CLASSES,
            stem_channels=cfg.MODEL.PointNet.stem_channels,
            local_channels=cfg.MODEL.PointNet.local_channels,
            global_channels=cfg.MODEL.PointNet.global_channels,
            dropout_prob=cfg.MODEL.PointNet.dropout_prob,
            with_transform=cfg.MODEL.PointNet.with_transform,
        )
        loss_fn = PointNetClsLoss(cfg.MODEL.PointNet.loss.reg_weight)
        metric = ClsAccuracy()
    elif cfg.TASK == 'part_segmentation':
        net = PointNetPartSeg(
            in_channels=cfg.INPUT.IN_CHANNELS,
            num_classes=cfg.DATASET.NUM_CLASSES,
            num_seg_classes=cfg.DATASET.NUM_SEG_CLASSES,
            stem_channels=cfg.MODEL.PointNet.stem_channels,
            local_channels=cfg.MODEL.PointNet.local_channels,
            cls_channels=cfg.MODEL.PointNet.cls_channels,
            seg_channels=cfg.MODEL.PointNet.seg_channels,
            dropout_prob_cls=cfg.MODEL.PointNet.dropout_prob_cls,
            dropout_prob_seg=cfg.MODEL.PointNet.dropout_prob_seg,
            with_transform=cfg.MODEL.PointNet.with_transform,
            use_one_hot=cfg.MODEL.PointNet.use_one_hot,
        )
        loss_fn = PointNetPartSegLoss(cfg.MODEL.PointNet.loss.reg_weight,
                                      cfg.MODEL.PointNet.loss.cls_weight,
                                      cfg.MODEL.PointNet.loss.seg_weight)
        metric = SegAccuracy()
    else:
        raise NotImplementedError()

    return net, loss_fn, metric
