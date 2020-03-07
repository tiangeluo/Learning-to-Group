"""Instance segmentation experiments configuration"""

from core.config.base import CN, _C

# public alias
cfg = _C

_C.TASK = 'ins_seg_3d'

_C.ITERATION_BASED = False

_C.TRAIN.VAL_METRIC = 'mAP'

# ----------------------------------------------------------------------------- #
# Input (Specific for point cloud)
# ----------------------------------------------------------------------------- #
# Input channels of point cloud
_C.INPUT.IN_CHANNELS = 3

# ----------------------------------------------------------------------------- #
# Label
# ----------------------------------------------------------------------------- #
_C.LABEL = CN()

# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #
_C.DATASET.PartNetInsSeg = CN()
_C.DATASET.PartNetInsSeg.TRAIN = CN(new_allowed=True)
_C.DATASET.PartNetInsSeg.TRAIN.split = 'train'
_C.DATASET.PartNetInsSeg.TRAIN.stage1 = 'fusion'
_C.DATASET.PartNetInsSeg.TRAIN.level = -1
_C.DATASET.PartNetInsSeg.VAL = CN()
_C.DATASET.PartNetInsSeg.VAL.split = 'val'
_C.DATASET.PartNetInsSeg.VAL.shape = ''
_C.DATASET.PartNetInsSeg.VAL.level = -1
_C.DATASET.PartNetInsSeg.TEST = CN()
_C.DATASET.PartNetInsSeg.TEST.split = 'test'
_C.DATASET.PartNetInsSeg.TEST.shape = ''
_C.DATASET.PartNetInsSeg.TEST.level = -1

_C.DATASET.PartNetRegionInsSeg = CN()
_C.DATASET.PartNetRegionInsSeg.TRAIN = CN(new_allowed=True)
_C.DATASET.PartNetRegionInsSeg.TRAIN.split = 'train'
_C.DATASET.PartNetRegionInsSeg.TRAIN.shape = ''
_C.DATASET.PartNetRegionInsSeg.TRAIN.stage1 = 'fusion'
_C.DATASET.PartNetRegionInsSeg.TRAIN.level = -1
_C.DATASET.PartNetRegionInsSeg.VAL = CN(new_allowed=True)
_C.DATASET.PartNetRegionInsSeg.VAL.split = 'val'
_C.DATASET.PartNetRegionInsSeg.VAL.shape = ''
_C.DATASET.PartNetRegionInsSeg.VAL.level = -1
_C.DATASET.PartNetRegionInsSeg.TEST = CN(new_allowed=True)
_C.DATASET.PartNetRegionInsSeg.TEST.split = 'test'
_C.DATASET.PartNetRegionInsSeg.TEST.shape = ''
_C.DATASET.PartNetRegionInsSeg.TEST.level = -1

_C.DATALOADER.KWARGS = CN(new_allowed=True)
_C.DATALOADER.KWARGS.num_centroids = 256
_C.DATALOADER.KWARGS.radius = 0.1
_C.DATALOADER.KWARGS.num_neighbours = 128
# _C.MODEL.PointNetInsSeg.with_renorm = False
# _C.MODEL.PointNetInsSeg.with_resample = False
# _C.MODEL.PointNetInsSeg.with_shift = False

_C.MODEL.CUSTOM = CN()
_C.MODEL.CUSTOM.classifier = CN()
_C.MODEL.CUSTOM.classifier.type = ''
_C.MODEL.CUSTOM.classifier.backbone = CN(new_allowed=True)
_C.MODEL.CUSTOM.classifier.head = CN(new_allowed=True)
_C.MODEL.CUSTOM.loss = CN()
_C.MODEL.CUSTOM.loss.ins_weight = 1.
#_C.MODEL.CUSTOM.loss.sampler = CN(new_allowed=True)


# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST.LEVEL = 3
_C.TEST.NUM_VOTES = 1

# relative path will be appended to OUTPUT_DIR
_C.TEST.SUBMIT_DIR = ''
