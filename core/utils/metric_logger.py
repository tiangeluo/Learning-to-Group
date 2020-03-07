# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
from __future__ import division
from collections import defaultdict
from collections import deque

import numpy as np
import torch


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.counts = deque(maxlen=window_size)
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.values.append(value)
        self.counts.append(count)
        self.sum += value
        self.count += count

    @property
    def avg(self):
        return np.sum(self.values) / np.sum(self.counts)

    @property
    def global_avg(self):
        return self.sum / self.count if self.count != 0 else float('nan')

    def reset(self):
        self.values.clear()
        self.counts.clear()
        self.sum = 0.0
        self.count = 0

    def __str__(self):
        return '{:.4f} ({:.4f})'.format(self.avg, self.global_avg)

    @property
    def summary_str(self):
        return '{:.4f}'.format(self.global_avg)


class Metric(AverageMeter):
    def __init__(self, *args, **kwargs):
        super(Metric, self).__init__(*args, **kwargs)
        self.training = True

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)

    def update_dict(self, preds, labels):
        """Update the metric

        Args:
            preds (dict): predictions
            labels (dict): labels or data
        """
        raise NotImplementedError()


class MetricList(object):
    """A list of metrics"""

    def __init__(self, metrics):
        self.metrics = metrics

    def train(self, mode=True):
        for metric in self.metrics:
            metric.train(mode)

    def eval(self):
        self.train(False)

    def update_dict(self, preds, labels):
        for metric in self.metrics:
            metric.update_dict(preds, labels)

    def __getitem__(self, item):
        return self.metrics[item]

    def __iter__(self):
        return iter(self.metrics)

    def reset(self):
        for metric in self.metrics:
            metric.reset()


class MetricLogger(object):
    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                count = v.numel()
                value = v.item() if count == 1 else v.sum().item()
            elif isinstance(v, np.ndarray):
                count = v.size
                value = v.item() if count == 1 else v.sum().item()
            else:
                assert isinstance(v, (float, int))
                value = v
                count = 1
            self.meters[k].update(value, count)

    def bind(self, metric_fn):
        if isinstance(metric_fn, MetricList):
            self.meters.update({x.name: x for x in metric_fn})
        elif isinstance(metric_fn, Metric):
            self.meters[metric_fn.name] = metric_fn
        else:
            raise TypeError('Unknown type of metric to bind: {}.'.format(type(metric_fn).__name__))

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        return getattr(self, attr)

    def __str__(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(metric_str)

    @property
    def summary_str(self):
        metric_str = []
        for name, meter in self.meters.items():
            metric_str.append('{}: {}'.format(name, meter.summary_str))
        return self.delimiter.join(metric_str)
