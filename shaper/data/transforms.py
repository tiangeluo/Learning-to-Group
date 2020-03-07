"""Helpers to transform point clouds for data augmentation.

Warnings:
    1. Be careful about in-place operations.
    2. x[indices] share the memory with x.
    3. Use random or torch.random instead of numpy.random
    since numpy will fork the same rng_state of the main process when using multi-processing.
    However, we have implemented a worker_init_fn to resolve this issue.

"""

import warnings
import random
import numpy as np
import torch


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def get_rotation_matrix_np(angle, axis):
    """Returns a 3x3 rotation matrix that performs a rotation around axis by angle
    Numpy version

    Args:
        angle (float or torch.Tensor): angle to rotate by
        axis (np.ndarray): axis to rotate about

    Returns:
        np.ndarray: 3x3 rotation matrix A. (y=A'x)

    References:
        https://en.wikipedia.org/wiki/Rotation_matrix

    """
    axis = np.asarray(axis)
    assert axis.ndim == 1 and axis.size == 3
    u = axis / np.linalg.norm(axis)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.cross(u, np.eye(3))
    R = cos_angle * np.eye(3) + sin_angle * cross_prod_mat + (1.0 - cos_angle) * np.outer(u, u)
    return R


def get_rotation_matrix_torch(angle, axis):
    """Return a rotation matrix by an angle around a given axis
    Pytorch version is slightly slower than numpy version.

    Args:
        angle (float): angle to rotate by
        axis (torch.Tensor): axis to rotate about. (3,)

    Returns:
        R (torch.Tensor): rotation matrix (3, 3) A. (y=A'x)

    References:
        https://en.wikipedia.org/wiki/Rotation_matrix

    """
    assert axis.numel() == 3
    u = axis / torch.norm(axis)
    u = u.view(1, -1)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    cross_product_matrix = torch.cross(u.expand(3, 3), torch.eye(3), dim=1)
    # Not necessary to transpose here
    R = cos_angle * torch.eye(3) + sin_angle * cross_product_matrix + (1 - cos_angle) * (u.t() @ u)
    return R


# ---------------------------------------------------------------------------- #
# Transformation only related to points
# ---------------------------------------------------------------------------- #
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points):
        for t in self.transforms:
            points = t(points)
        return points

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    def __init__(self):
        warnings.warn('ToTensor is an in-place operator. Please be very careful when using it!')

    def __call__(self, points):
        assert isinstance(points, np.ndarray)
        # Since it is tricky to decide whether to copy data, we leave it to dataset.
        # Note that torch.tensor always copies data, while torch.as_tensor does not.
        return torch.as_tensor(points, dtype=torch.float32)


class Transpose(object):
    def __call__(self, points):
        if isinstance(points, np.ndarray):
            return points.transpose()
        elif isinstance(points, torch.Tensor):
            return points.transpose_(0, 1)
        else:
            raise TypeError('Wrong type {} to transpose.'.format(type(points).__name__))


class Rotate(object):
    """Rotate along an axis by a random angle"""

    def __init__(self, axis):
        self.axis = np.asarray(axis)

    def _get_rotation_matrix(self):
        # angle = random.uniform(0, 2 * np.pi)
        # angle = np.random.rand() * (2 * np.pi)
        angle = torch.rand(1).item() * (2 * np.pi)
        return get_rotation_matrix_np(angle, self.axis)

    def __call__(self, points):
        rotation_matrix = points.new_tensor(self._get_rotation_matrix())
        points[:, 0:3] = points[:, 0:3] @ rotation_matrix
        return points


class RotateWithNormal(Rotate):
    """Rotate along an axis by a random angle with normals"""

    def __call__(self, points):
        rotation_matrix = points.new_tensor(self._get_rotation_matrix())
        assert points.size(1) >= 6, 'Expect dimension >= 6, but get {:d}'.format(points.size(1))
        points[:, 0:3] = points[:, 0:3] @ rotation_matrix
        points[:, 3:6] = points[:, 3:6] @ rotation_matrix
        return points


class RotateY(Rotate):
    """ModelNet and ShapeNetPart is y-axis upward."""

    def __init__(self):
        super(RotateY, self).__init__((0., 1., 0.))


class RotateYWithNormal(RotateWithNormal):
    """ModelNet and ShapeNetPart is y-axis upward."""

    def __init__(self):
        super(RotateYWithNormal, self).__init__((0., 1., 0.))


class RotateByAngle(object):
    _AXES = {
        'x': (1.0, 0.0, 0.0),
        'y': (0.0, 1.0, 0.0),
        'z': (0.0, 0.0, 1.0),
    }

    def __init__(self, axis, angle):
        if axis in self._AXES:
            axis = self._AXES[axis]
        self.axis = np.asarray(axis)
        self.rotation_matrix = get_rotation_matrix_np(angle, self.axis)
        self.rotation_matrix = torch.as_tensor(self.rotation_matrix, dtype=torch.float32)

    def __call__(self, points):
        points[:, 0:3] = points[:, 0:3] @ self.rotation_matrix
        return points


class RotateByAngleWithNormal(RotateByAngle):
    def __call__(self, points):
        points[:, 0:3] = points[:, 0:3] @ self.rotation_matrix
        points[:, 3:6] = points[:, 3:6] @ self.rotation_matrix
        return points


class RotatePerturbation(object):
    """Small perturbation along three axes"""

    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip
        self.axes = np.eye(3)

    def _get_rotation_matrix(self):
        # angles = np.clip([random.gauss(0, self.angle_sigma) for _ in range(3)], -self.angle_clip, self.angle_clip)
        # angles = np.clip(np.random.randn(3) * self.angle_sigma, -self.angle_clip, self.angle_clip)
        angles = torch.clamp(self.angle_sigma * torch.randn(3), -self.angle_clip, self.angle_clip).numpy()

        Rx = get_rotation_matrix_np(angles[0], self.axes[0])
        Ry = get_rotation_matrix_np(angles[1], self.axes[1])
        Rz = get_rotation_matrix_np(angles[2], self.axes[2])
        return Rz @ Ry @ Rx

    def __call__(self, points):
        rotation_matrix = points.new_tensor(self._get_rotation_matrix())
        points[:, 0:3] = points[:, 0:3] @ rotation_matrix
        return points


class RotatePerturbationWithNormal(RotatePerturbation):
    def __call__(self, points):
        rotation_matrix = points.new_tensor(self._get_rotation_matrix())
        points[:, 0:3] = points[:, 0:3] @ rotation_matrix
        points[:, 3:6] = points[:, 3:6] @ rotation_matrix
        return points


class Translate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        # translation = np.random.uniform(-self.translate_range, self.translate_range, (3,))
        translation = points.new(3).uniform_(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation  # broadcast first dimension
        return points


class Scale(object):
    """Scale should be applied before Translate, otherwise the aspect ratio will be distorted."""

    def __init__(self, lo=0.8, hi=1.25):
        assert hi >= lo
        self.lo = lo
        self.hi = hi

    def __call__(self, points):
        # scale = random.uniform(self.lo, self.hi)
        scale = points.new(1).uniform_(self.lo, self.hi)
        points[:, 0:3] *= scale
        return points


class Jitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = points.new(points.size(0), 3).normal_(mean=0.0, std=self.std)
        jittered_data = jittered_data.clamp_(-self.clip, self.clip)
        points[:, 0:3] += jittered_data
        return points


# ---------------------------------------------------------------------------- #
# Transformation related to points and point-wise labels
# ---------------------------------------------------------------------------- #
class ComposeSeg(Compose):
    def __call__(self, points, seg_label):
        for t in self.transforms:
            if isinstance(t, (Shuffle, RandomDropout, Sample)):
                points, seg_label = t(points, seg_label)
            else:
                points = t(points)
        return points, seg_label


class Shuffle(object):
    def __init__(self):
        super(Shuffle, self).__init__()

    def __call__(self, points, seg_label=None):
        index = torch.randperm(points.size(0))
        if seg_label is None:
            return points[index]
        else:
            # Note that even if seg_label is numpy.ndarray, it still works.
            return points[index], seg_label[index]


class RandomDropout(object):
    def __init__(self, max_dropout_ratio=0.875, ignore_index=-100):
        assert 0 <= max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.ignore_index = ignore_index

    def __call__(self, points, seg_label=None):
        dropout_ratio = torch.rand(1) * self.max_dropout_ratio
        dropout_indices = torch.nonzero(torch.rand(points.size(0)) <= dropout_ratio)[:, 0]
        is_drop = dropout_indices.numel() > 0
        if seg_label is None:
            if is_drop:
                points[dropout_indices] = points[0]  # set to the first point
            return points
        else:
            if is_drop:
                points[dropout_indices] = points[0]
                # ignore the labels of duplicated points
                seg_label[dropout_indices] = self.ignore_index
            return points, seg_label


class Sample(object):
    def __init__(self, num_points):
        self.num_points = num_points

    def __call__(self, points, seg_label=None):
        choice = torch.randint(points.size(0), size=(self.num_points,), dtype=torch.int64)
        if seg_label is None:
            return points[choice]
        else:
            return points[choice], seg_label[choice]
