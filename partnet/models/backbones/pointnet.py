"""PointNet for part segmentation

References:
    @article{qi2016pointnet,
      title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
      author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
      journal={arXiv preprint arXiv:1612.00593},
      year={2016}
    }
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn import MLP, SharedMLP, Conv1d
from core.nn.init import xavier_uniform, set_bn
from shaper.models.pointnet.pointnet_cls import TNet
from IPython import embed


class Stem(nn.Module):
    """Stem (main body or stalk). Extract features from raw point clouds"""

    def __init__(self,
                 in_channels,
                 stem_channels=(64, 128, 128),
                 with_transform=True):
        super(Stem, self).__init__()

        self.in_channels = in_channels
        self.out_channels = stem_channels[-1]
        self.with_transform = with_transform

        # feature stem
        self.mlp = SharedMLP(in_channels, stem_channels)
        self.mlp.reset_parameters(xavier_uniform)

        if self.with_transform:
            # input transform
            self.transform_input = TNet(in_channels, in_channels)
            # feature transform
            self.transform_feature = TNet(self.out_channels, self.out_channels)

    def forward(self, x):
        """PointNet Stem forward

        Args:
            x (torch.Tensor): (batch_size, in_channels, num_points)

        Returns:
            torch.Tensor: (batch_size, stem_channels[-1], num_points)
            dict (optional):
                trans_input: (batch_size, in_channels, in_channels)
                trans_feature: (batch_size, stem_channels[-1], stem_channels[-1])
                stem_features (list of torch.Tensor)

        """
        end_points = {}

        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # feature
        features = []
        for module in self.mlp:
            x = module(x)
            features.append(x)
        end_points['stem_features'] = features

        # feature transform
        if self.with_transform:
            trans_feature = self.transform_feature(x)
            x = torch.bmm(trans_feature, x)
            end_points['trans_feature'] = trans_feature

        return x, end_points


class PointNet(nn.Module):
    """PointNet for part segmentation

     Args:
        in_channels (int): the number of input channels
        stem_channels (tuple of int): the numbers of channels in stem feature extractor
        local_channels (tuple of int): the numbers of channels in local mlp
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob_seg (float): the probability to dropout in segmentation mlp

    References:
        https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py

    """

    def __init__(self,
                 in_channels,
                 stem_channels=(16, 32, 32),
                 local_channels=(128, 128),
                 seg_channels=(64, 64, 32),
                 dropout_prob_seg=0.2):
        super(PointNet, self).__init__()

        self.in_channels = in_channels

        # stem
        self.stem = Stem(in_channels, stem_channels, with_transform=False)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels)

        # part segmentation
        # Notice that the original repo concatenates global feature, one hot class embedding,
        # stem features and local features. However, the paper does not use last local feature.
        # Here, we follow the released repo.
        in_channels_seg = sum(stem_channels) + sum(local_channels) + local_channels[-1]
        self.mlp_seg = SharedMLP(in_channels_seg, seg_channels[:-1], dropout_prob=dropout_prob_seg)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1)

        self.reset_parameters()

    def forward(self, points):
        x = points
        num_points = x.shape[2]
        end_points = {}

        # stem
        stem_feature, end_points_stem = self.stem(x)
        stem_features = end_points_stem.pop('stem_features')
        end_points.update(end_points_stem)

        # mlp for local features
        local_features = []
        x = stem_feature
        for ind, mlp in enumerate(self.mlp_local):
            x = mlp(x)
            local_features.append(x)

        # max pool over points
        global_feature, max_indices = torch.max(x, 2)  # (batch_size, local_channels[-1])
        # end_points['key_point_indices'] = max_indices

        # segmentation
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, num_points)
        seg_features = stem_features + local_features + [global_feature_expand]

        x = torch.cat(seg_features, dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)

        preds = {
            'feature': x
        }
        preds.update(end_points)

        return preds

    def reset_parameters(self):
        # default initialization
        self.mlp_local.reset_parameters(xavier_uniform)
        self.mlp_seg.reset_parameters(xavier_uniform)
        self.conv_seg.reset_parameters(xavier_uniform)
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)
