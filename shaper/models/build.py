"""Build models

Notes:
    When a new model is implemented, please provide a builder to build the model with config,
    and register it in _MODEL_BUILDERS

    How to implement a model:
    1. Modularize the model
    2. Try to add in_channels, out_channels to all the modules' attributes
    3. For the complete model, like PointNetCls, output a non-nested dictionary instead of single tensor or tuples
    4. Implement loss module whose inputs are preds and labels. Both of inputs are dict.
    5. Implement metric module (or use a general module in 'metric.py')

"""

from .pointnet.build import build_pointnet
from .pointnet2.build import build_pointnet2ssg, build_pointnet2msg
from .dgcnn.build import build_dgcnn

_MODEL_BUILDERS = {
    'PointNet': build_pointnet,
    'PN2SSG': build_pointnet2ssg,
    'PN2MSG': build_pointnet2msg,
    'DGCNN': build_dgcnn,
}


def build_model(cfg):
    return _MODEL_BUILDERS[cfg.MODEL.TYPE](cfg)


def register_model_builder(name, builder):
    if name in _MODEL_BUILDERS:
        raise KeyError(
            'Duplicate keys for {:s} with {} and {}.'
            'Solve key conflicts first!'.format(name, _MODEL_BUILDERS[name], builder))
    _MODEL_BUILDERS[name] = builder
