"""Open3D visualization tools

References:
    @article{Zhou2018,
        author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
        title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
        journal   = {arXiv:1801.09847},
        year      = {2018},
    }

See Also:
    https://github.com/IntelVCL/Open3D
    https://github.com/IntelVCL/Open3D-PointNet/blob/master/open3d_visualilze.py

"""

import open3d


def draw_point_cloud(points, colors=None, normals=None):
    pc = open3d.PointCloud()
    pc.points = open3d.Vector3dVector(points)
    if colors is not None:
        pc.colors = open3d.Vector3dVector(colors)
    if normals is not None:
        pc.normals = open3d.Vector3dVector(normals)
    return pc


def visualize_point_cloud(points, colors=None, normals=None):
    pc = draw_point_cloud(points, colors, normals)
    mesh_frame = open3d.create_mesh_coordinate_frame(size=1.0, origin=[0, 0, 0])
    open3d.draw_geometries([pc, mesh_frame])
