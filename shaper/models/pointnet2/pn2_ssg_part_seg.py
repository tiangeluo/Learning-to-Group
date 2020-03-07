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


class PointNet2SSGPartSeg(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        in_channels (int): the number of input channels
        num_classes (int): the number of classification class
        num_seg_classes (int): the number of segmentation class
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features
        use_xyz (bool): whether or not to use the xyz position of a points as a feature
        use_one_hot (bool): whehter to use one hot vector of class labels.

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_seg_classes,
                 num_centroids=(512, 128, 0),
                 radius=(0.2, 0.4, -1.0),
                 num_neighbours=(64, 64, -1),
                 sa_channels=((64, 64, 128), (128, 128, 256), (256, 512, 1024)),
                 fp_channels=((256, 256), (256, 128), (128, 128, 128)),
                 num_fp_neighbours=(0, 3, 3),
                 seg_channels=(128,),
                 dropout_prob=0.5,
                 use_xyz=True,
                 use_one_hot=True):
        super(PointNet2SSGPartSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.use_xyz = use_xyz
        self.use_one_hot = use_one_hot

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
        if self.use_one_hot:
            inter_channels[0] += num_classes  # concat with one-hot
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
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        self.reset_parameters()

    def forward(self, data_batch):
        points = data_batch['points']
        end_points = {}

        xyz = points.narrow(1, 0, 3)
        if points.size(1) > 3:
            feature = points.narrow(1, 3, points.size(1) - 3)
        else:
            feature = None

        # save intermediate results
        inter_xyz = [xyz]
        inter_feature = [points if self.use_xyz else feature]

        if self.use_one_hot:
            # one hot class label
            num_points = points.size(2)
            with torch.no_grad():
                cls_label = data_batch['cls_label']
                one_hot = cls_label.new_zeros(cls_label.size(0), self.num_classes)
                one_hot = one_hot.scatter(1, cls_label.unsqueeze(1), 1)  # (batch_size, num_classes)
                one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points).float()
                inter_feature[0] = torch.cat((inter_feature[0], one_hot_expand), dim=1)

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
        seg_logit = self.seg_logit(x)

        preds = {
            'seg_logit': seg_logit,
        }
        preds.update(end_points)

        return preds

    def reset_parameters(self):
        for sa_module in self.sa_modules:
            sa_module.reset_parameters(xavier_uniform)
        for fp_module in self.fp_modules:
            fp_module.reset_parameters(xavier_uniform)
        self.mlp_seg.reset_parameters(xavier_uniform)
        xavier_uniform(self.seg_logit)
        set_bn(self, momentum=0.01)


def test_PointNet2SSGPartSeg():
    batch_size = 8
    in_channels = 6
    num_points = 2048
    num_classes = 16
    num_seg_classes = 50

    points = torch.randn(batch_size, in_channels, num_points)
    points = points.cuda()
    cls_label = torch.randint(num_classes, (batch_size,))
    cls_label = cls_label.cuda()

    pn2ssg = PointNet2SSGPartSeg(in_channels, num_classes, num_seg_classes)
    pn2ssg.cuda()
    print(pn2ssg)
    out_dict = pn2ssg({'points': points, 'cls_label': cls_label})
    for k, v in out_dict.items():
        print('PointNet2SSG:', k, v.shape)
