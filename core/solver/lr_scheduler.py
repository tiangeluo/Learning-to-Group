from __future__ import division
from bisect import bisect_right

from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR


class WarmupMultiStepLR(MultiStepLR):
    """Set the learning rate with linear warm-up"""
    def __init__(self, optimizer, milestones, warmup_step, warmup_gamma, gamma=0.1, last_epoch=-1):
        self.warmup_step = warmup_step
        self.warmup_gamma = warmup_gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_step:
            return [base_lr * (self.warmup_gamma + self.last_epoch / self.warmup_step * (1 - self.warmup_gamma))
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch - self.warmup_step)
                    for base_lr in self.base_lrs]


class ClipLR(object):
    """Clip the learning rate of a given scheduler.
    Same interfaces of _LRScheduler should be implemented.

    Args:
        scheduler (_LRScheduler): an instance of _LRScheduler.
        min (float): minimum learning rate.

    """
    def __init__(self, scheduler, min=1e-5):
        self.scheduler = scheduler
        self.min = min

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        return [max(self.min, lr) for lr in self.scheduler.get_lr()]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.scheduler.last_epoch + 1
        self.scheduler.last_epoch = epoch
        for param_group, lr in zip(self.scheduler.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def __getattr__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        else:
            return getattr(self.scheduler, item)
