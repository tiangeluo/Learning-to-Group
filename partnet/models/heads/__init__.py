from .concat_head import ConcatHead


def build_head(head_cfg):
    obj_type = head_cfg.pop('type')
    return eval(obj_type)(**head_cfg)
