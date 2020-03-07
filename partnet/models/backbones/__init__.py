from .pointnet import PointNet
from .pn2_ssg import PointNet2SSG


def build_backbone(backbone_cfg):
    obj_type = backbone_cfg.pop('type')
    return eval(obj_type)(**backbone_cfg)

