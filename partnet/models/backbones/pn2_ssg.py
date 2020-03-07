"""PointNet++

References:
    @article{qi2017pointnetplusplus,
      title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
      author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1706.02413},
      year={2017}
    }
"""

import torch
import torch.nn as nn

from core.nn import SharedMLP
from core.nn.init import xavier_uniform, set_bn
from shaper.models.pointnet2.modules import PointNetSAModule, PointnetFPModule


class PointNet2SSG(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        in_channels (int): the number of input channels
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features
        use_xyz (bool): whether or not to use the xyz position of a points as a feature

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """

    def __init__(self,
                 in_channels,
                 num_centroids=(128, 32, 0),
                 radius=(0.2, 0.4, -1.0),
                 num_neighbours=(64, 64, -1),
                 sa_channels=((16, 16, 32), (32, 32, 64), (128, 128, 256)),
                 fp_channels=((64, 64), (64, 32), (32, 32, 32)),
                 num_fp_neighbours=(0, 3, 3),
                 seg_channels=(32,),
                 dropout_prob=0.5,
                 use_xyz=True):
        super(PointNet2SSG, self).__init__()

        self.in_channels = in_channels
        self.use_xyz = use_xyz

        # Sanity check
        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)
        assert len(radius) == num_sa_layers
        assert len(num_neighbours) == num_sa_layers
        assert len(sa_channels) == num_sa_layers
        assert num_sa_layers == num_fp_layers
        assert len(num_fp_neighbours) == num_fp_layers

        # Set Abstraction Layers
        feature_channels = in_channels - 3
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = PointNetSAModule(in_channels=feature_channels,
                                         mlp_channels=sa_channels[ind],
                                         num_centroids=num_centroids[ind],
                                         radius=radius[ind],
                                         num_neighbours=num_neighbours[ind],
                                         use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [in_channels if use_xyz else in_channels - 3]
        inter_channels.extend([x[-1] for x in sa_channels])

        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            fp_module = PointnetFPModule(in_channels=feature_channels + inter_channels[-2 - ind],
                                         mlp_channels=fp_channels[ind],
                                         num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        self.mlp_seg = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)

        self.reset_parameters()

    def forward(self, points):
        end_points = {}

        xyz = points.narrow(1, 0, 3)
        if points.size(1) > 3:
            feature = points.narrow(1, 3, points.size(1) - 3)
        else:
            feature = None

        # save intermediate results
        inter_xyz = [xyz]
        inter_feature = [points if self.use_xyz else feature]

        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

        # MLP
        x = self.mlp_seg(sparse_feature)

        preds = {
            'feature': x,
        }
        preds.update(end_points)

        return preds

    def reset_parameters(self):
        for sa_module in self.sa_modules:
            sa_module.reset_parameters(xavier_uniform)
        for fp_module in self.fp_modules:
            fp_module.reset_parameters(xavier_uniform)
        self.mlp_seg.reset_parameters(xavier_uniform)
        set_bn(self, momentum=0.01)
