from .region_classifier import RegionClassifier


def build_classifier(classifer_cfg):
    obj_type = classifer_cfg.pop('type')
    return eval(obj_type)(**classifer_cfg)
