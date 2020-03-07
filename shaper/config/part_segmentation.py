"""Part segmentation experiments configuration"""

from core.config.base import CN, _C

# public alias
cfg = _C

_C.TASK = "part_segmentation"

_C.TRAIN.VAL_METRIC = "seg_acc"

# ---------------------------------------------------------------------------- #
# INPUT (Specific for point cloud)
# ---------------------------------------------------------------------------- #
# Input channels of point cloud
# If channels == 3, (x, y, z)
# If channels == 6: (x, y, z, normal_x, normal_y, normal_z)
_C.INPUT.IN_CHANNELS = 3
# -1 for all points
_C.INPUT.NUM_POINTS = -1
# Whether to use normal. Assume points[.., 3:6] is normal.
_C.INPUT.USE_NORMAL = False

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
_C.DATASET.NUM_SEG_CLASSES = 0

# ---------------------------------------------------------------------------- #
# Test-time augmentations for point cloud classification
# ---------------------------------------------------------------------------- #
_C.TEST.VOTE = CN()

_C.TEST.VOTE.NUM_VOTE = 0

_C.TEST.VOTE.TYPE = ""

# Multi-view voting
_C.TEST.VOTE.MULTI_VIEW = CN()
# The axis along which to rotate
_C.TEST.VOTE.MULTI_VIEW.AXIS = "y"

# Data augmentation, different with TEST.AUGMENTATION.
# Use for voting only
_C.TEST.VOTE.AUGMENTATION = ()

# Whether to shuffle points from different views (especially for methods like PointNet++)
_C.TEST.VOTE.SHUFFLE = False

# ---------------------------------------------------------------------------- #
# PointNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.PointNet = CN()

_C.MODEL.PointNet.stem_channels = (64, 128, 128)
_C.MODEL.PointNet.local_channels = (512, 2048)
_C.MODEL.PointNet.cls_channels = (256, 256)
_C.MODEL.PointNet.seg_channels = (256, 256, 128)

_C.MODEL.PointNet.dropout_prob_cls = 0.3
_C.MODEL.PointNet.dropout_prob_seg = 0.2
_C.MODEL.PointNet.with_transform = True
_C.MODEL.PointNet.use_one_hot = True

_C.MODEL.PointNet.loss = CN()
_C.MODEL.PointNet.loss.reg_weight = 0.032
_C.MODEL.PointNet.loss.cls_weight = 0.0
_C.MODEL.PointNet.loss.seg_weight = 1.0

# ---------------------------------------------------------------------------- #
# PN2SSG options
# ---------------------------------------------------------------------------- #
_C.MODEL.PN2SSG = CN()

_C.MODEL.PN2SSG.num_centroids = (512, 128, 0)
_C.MODEL.PN2SSG.radius = (0.2, 0.4, -1.0)
_C.MODEL.PN2SSG.num_neighbours = (32, 64, -1)
_C.MODEL.PN2SSG.sa_channels = ((64, 64, 128), (128, 128, 256), (256, 512, 1024))
_C.MODEL.PN2SSG.fp_channels = ((256, 256), (256, 128), (128, 128, 128))
_C.MODEL.PN2SSG.num_fp_neighbours = (0, 3, 3)
_C.MODEL.PN2SSG.seg_channels = (128,)
_C.MODEL.PN2SSG.dropout_prob = 0.5
_C.MODEL.PN2SSG.use_xyz = True
_C.MODEL.PN2SSG.use_one_hot = True

# ---------------------------------------------------------------------------- #
# PN2MSG options
# ---------------------------------------------------------------------------- #
_C.MODEL.PN2MSG = CN()

_C.MODEL.PN2MSG.num_centroids = (512, 128, 0)
_C.MODEL.PN2MSG.radius_list = ((0.1, 0.2, 0.4), (0.4, 0.8), -1.0)
_C.MODEL.PN2MSG.num_neighbours_list = ((32, 64, 128), (64, 128), -1)
_C.MODEL.PN2MSG.sa_channels_list = (
    ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
    ((128, 128, 256), (128, 196, 256)),
    (256, 512, 1024),
)
_C.MODEL.PN2MSG.fp_channels = ((256, 256), (256, 128), (128, 128))
_C.MODEL.PN2MSG.num_fp_neighbours = (0, 3, 3)
_C.MODEL.PN2MSG.seg_channels = (128,)
_C.MODEL.PN2MSG.dropout_prob = 0.5
_C.MODEL.PN2MSG.use_xyz = True
_C.MODEL.PN2MSG.use_one_hot = True

# ---------------------------------------------------------------------------- #
# DGCNN options
# ---------------------------------------------------------------------------- #
_C.MODEL.DGCNN = CN()

_C.MODEL.DGCNN.edge_conv_channels = ((64, 64), (64, 64), 64)
_C.MODEL.DGCNN.local_channels = (1024,)
_C.MODEL.DGCNN.seg_channels = (256, 256, 128)
_C.MODEL.DGCNN.k = 20

_C.MODEL.DGCNN.dropout_prob = 0.4
_C.MODEL.DGCNN.with_transform = True
