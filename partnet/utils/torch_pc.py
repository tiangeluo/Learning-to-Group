import torch


def normalize_points(points):
    """Normalize point cloud

    Args:
        points (torch.Tensor): (batch_size, num_points, 3)

    Returns:
        torch.Tensor: normalized points

    """
    assert points.dim() == 3 and points.size(2) == 3
    centroid = points.mean(dim=1, keepdim=True)
    points = points - centroid
    norm, _ = points.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)
    new_points = points / norm
    return new_points


def select_points(points, index):
    """Gather xyz of centroids according to indices

    Args:
        points: (batch_size, channels, num_points)
        index: (batch_size, num_centroids)

    Returns:
        new_xyz (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = points.size(0)
    channels = points.size(1)
    num_centroids = index.size(1)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids)
    return points.gather(2, index_expand)


def group_points(points, index):
    """Gather points by index

    Args:
        points (torch.Tensor): (batch_size, channels, num_points)
        index (torch.Tensor): (batch_size, num_centroids, num_neighbours), indices of neighbours of each centroid.

    Returns:
        group_points (torch.Tensor): (batch_size, channels, num_centroids, num_neighbours), grouped points.

    """
    batch_size = points.size(0)
    channels = points.size(1)
    num_points = points.size(2)
    num_centroids = index.size(1)
    num_neighbours = index.size(2)
    # (batch_size, channels, num_centroids, num_points)
    points_expand = points.unsqueeze(2).expand(batch_size, channels, num_centroids, num_points)
    # (batch_size, channels, num_centroids, num_neighbours)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids, num_neighbours)

    # (batch_size, channels, num_centroids, num_neighbours)
    output = points_expand.gather(3, index_expand)
    return output


