import time
import os
import os.path as osp

from .metric_logger import AverageMeter
from tensorboardX import SummaryWriter

_KEYWORDS = ('loss', 'acc', 'IoU')


class TensorboardLogger(object):
    def __init__(self, log_dir, keywords=_KEYWORDS):
        self.log_dir = osp.join(log_dir, 'events.{}'.format(time.strftime('%m-%d_%H-%M-%S')))
        os.makedirs(self.log_dir, exist_ok=True)
        self.keywords = keywords
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def add_scalars(self, meters, step, prefix=''):
        for name, meter in meters.items():
            if all(keyword not in name for keyword in self.keywords):
                continue
            if isinstance(meter, AverageMeter):
                v = meter.global_avg
            elif isinstance(meter, (int, float)):
                v = meter
            else:
                raise TypeError('Unknown meter type {}.'.format(type(meter).__name__))
            if isinstance(v, dict):
                for suffix, value in v.items():
                    self.writer.add_scalar(osp.join(prefix, '{}_{}'.format(name, suffix)), value, global_step=step)
            else:
                self.writer.add_scalar(osp.join(prefix, name), v, global_step=step)
