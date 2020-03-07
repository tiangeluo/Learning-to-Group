import numpy as np
import torch

from shaper.data import transforms as T
from shaper.utils.pc_util import normalize_points


def get_Rx(angle):
    return np.asarray([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)],
    ])


def get_Rz(angle):
    return np.asarray([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])


def test_rotation_matrix():
    angle = np.random.uniform(0, 2 * np.pi)
    axis = np.random.normal(0, 1.0, (3,)).astype(np.float32)

    # Numpy version
    R = T.get_rotation_matrix_np(angle=angle, axis=(0.0, 0.0, 1.0))
    np.testing.assert_allclose(R, get_Rz(angle))
    R = T.get_rotation_matrix_np(angle=angle, axis=(1.0, 0.0, 0.0))
    np.testing.assert_allclose(R, get_Rx(angle))

    # Pytorch version
    R1 = T.get_rotation_matrix_np(angle=angle, axis=axis).astype(np.float32)
    R2 = T.get_rotation_matrix_torch(angle=torch.as_tensor(angle), axis=torch.as_tensor(axis))
    np.testing.assert_allclose(R1, R2.numpy(), atol=1e-6)


def test_transform():
    import open3d
    from shaper.utils.visualize import draw_point_cloud, visualize_point_cloud

    x, y, z = np.meshgrid(np.arange(10), np.arange(10), np.arange(10), indexing='xy')
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    points = normalize_points(points)
    normals = points.copy()  # dummy normals
    pc = np.hstack([points, normals]).astype(np.float32)
    # visualize_point_cloud(points, normals=normals)

    old_colors = np.ones_like(points) * (255, 0, 0)
    new_colors = np.ones_like(points) * (0, 255, 0)
    # to_tensor = T.ToTensor()
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)

    # rotate by angle
    rotate_z = T.RotateByAngleWithNormal(axis='z', angle=np.pi/4)
    rotated_points = rotate_z(to_tensor(pc)).numpy()
    visualize_point_cloud(rotated_points[:, 0:3], normals=rotated_points[:, 3:6])

    # rotate
    rotate = T.RotatePerturbationWithNormal(1.0, 1.0)
    rotated_points = rotate(to_tensor(pc)).numpy()
    visualize_point_cloud(rotated_points[:, 0:3], normals=rotated_points[:, 3:6])

    # translate
    translate = T.Translate()
    translated_points = translate(to_tensor(pc)).numpy()
    open3d.draw_geometries([draw_point_cloud(translated_points[:, 0:3], new_colors, translated_points[:, 3:6]),
                            draw_point_cloud(points, old_colors)])

    # scale
    scale = T.Scale(2.0, 3.0)
    scaled_points = scale(to_tensor(pc)).numpy()
    open3d.draw_geometries([draw_point_cloud(scaled_points[:, 0:3], new_colors, scaled_points[:, 3:6]),
                            draw_point_cloud(points, old_colors)])

    # jitter
    jitter = T.Jitter()
    jittered_points = jitter(to_tensor(pc)).numpy()
    open3d.draw_geometries([draw_point_cloud(jittered_points[:, 0:3], new_colors, jittered_points[:, 3:6]),
                            draw_point_cloud(points, old_colors)])

    # random dropout
    random_dropout = T.RandomDropout(0.99)
    dropout_points = random_dropout(to_tensor(np.hstack([pc, new_colors]))).numpy()
    visualize_point_cloud(dropout_points[:, 0:3], dropout_points[:, 6:9], dropout_points[:, 3:6])
