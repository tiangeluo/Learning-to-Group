import numpy as np


def normalize_points(points):
    """Normalize point cloud

    Args:
        points (np.ndarray): (n, 3)

    Returns:
        np.ndarray: normalized points

    """
    assert points.ndim == 2 and points.shape[1] == 3
    centroid = np.mean(points, axis=0)
    points = points - centroid
    norm = np.max(np.linalg.norm(points, ord=2, axis=1))
    new_points = points / norm
    return new_points


def crop_or_pad_points(points, num_points=-1, shuffle=False):
    """Crop or pad point cloud to a fixed number

    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order

    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points

    """
    if shuffle:
        choice = np.random.permutation(len(points))
    else:
        choice = np.arange(len(points))

    if num_points > 0:
        if len(points) >= num_points:
            choice = choice[:num_points]
        else:
            num_pad = num_points - len(points)
            pad = np.random.choice(choice, num_pad, replace=True)
            choice = np.concatenate([choice, pad])

    # Pad with replacement (used in original PointNet and PointNet++)
    # choice = np.random.choice(len(points), num_points, replace=True)

    # Return a copy to avoid operating original data
    new_points = points[choice].copy()

    return new_points, choice


def pad_points(points, num_points=-1, random=False):
    """Crop or pad point cloud to a fixed number

    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        random (bool): whether to pad with a random subset or the first point.

    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points

    """
    choice = np.arange(len(points))
    if num_points > 0 and num_points > len(choice):
        num_pad = num_points - len(points)
        if random:
            pad = np.random.choice(choice, num_pad, replace=True)
        else:
            pad = np.zeros(num_pad, dtype=choice.dtype)
        choice = np.concatenate([choice, pad])

    # Return a copy to avoid corrupting original data
    new_points = points[choice].copy()

    return new_points, choice
