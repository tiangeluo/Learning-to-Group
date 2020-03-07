import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from core.solver.lr_scheduler import WarmupMultiStepLR, ClipLR


def test_WarmupMultiStepLR():
    target = [0.5, 0.75] + [1.0] * 5 + [0.1] * 5
    optimizer = torch.optim.SGD([torch.nn.Parameter()], lr=1.0)
    lr_scheduler = WarmupMultiStepLR(optimizer, milestones=[5], warmup_step=2, warmup_gamma=0.5, gamma=0.1)
    output = []
    for epoch in range(2 + 10):
        lr_scheduler.step()
        output.extend(lr_scheduler.get_lr())
    np.testing.assert_allclose(output, target, atol=1e-6)


def test_ClipLR():
    target = [0.1 ** i for i in range(4)] + [1e-3]
    optimizer = torch.optim.SGD([torch.nn.Parameter()], lr=1.0)
    lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler = ClipLR(lr_scheduler, min=1e-3)
    output = []
    for epoch in range(5):
        lr_scheduler.step()
        output.extend(lr_scheduler.get_lr())
    np.testing.assert_allclose(output, target, atol=1e-6)
