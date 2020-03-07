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

from core.nn import MLP
from core.nn.init import xavier_uniform, set_bn
from shaper.models.pointnet2.modules import PointNetSAModule, PointNetSAModuleMSG


class PointNet2MSGCls(nn.Module):
    """PointNet2 with multi-scale grouping for classification"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_centroids=(512, 128, 0),
                 radius_list=((0.1, 0.2, 0.4), (0.2, 0.4, 0.8), -1.0),
                 num_neighbours_list=((16, 32, 128), (32, 64, 128), -1),
                 sa_channels_list=(
                         ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
                         ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
                         (256, 512, 1024),
                 ),
                 global_channels=(512, 256),
                 dropout_prob=0.5,
                 use_xyz=True):
        super(PointNet2MSGCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_xyz = use_xyz

        # sanity check
        num_sa_layers = len(num_centroids)
        assert len(radius_list) == num_sa_layers
        assert len(num_neighbours_list) == num_sa_layers
        assert len(sa_channels_list) == num_sa_layers

        feature_channels = in_channels - 3
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers - 1):
            sa_module = PointNetSAModuleMSG(in_channels=feature_channels,
                                            mlp_channels_list=sa_channels_list[ind],
                                            num_centroids=num_centroids[ind],
                                            radius_list=radius_list[ind],
                                            num_neighbours_list=num_neighbours_list[ind],
                                            use_xyz=use_xyz)
            self.sa_modules.append(sa_module)
            feature_channels = sa_module.out_channels

        sa_module = PointNetSAModule(in_channels=feature_channels,
                                     mlp_channels=sa_channels_list[-1],
                                     num_centroids=num_centroids[-1],
                                     radius=radius_list[-1],
                                     num_neighbours=num_neighbours_list[-1],
                                     use_xyz=use_xyz)
        self.sa_modules.append(sa_module)

        self.mlp_global = MLP(sa_channels_list[-1][-1], global_channels, dropout_prob=dropout_prob)
        self.classifier = nn.Linear(global_channels[-1], out_channels, bias=True)

        self.reset_parameters()

    def forward(self, data_batch):
        point = data_batch['points']
        end_points = {}

        # torch.Tensor.narrow; share same memory
        xyz = point.narrow(1, 0, 3)
        if point.size(1) > 3:
            feature = point.narrow(1, 3, point.size(1) - 3)
        else:
            feature = None

        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)

        x, max_indices = torch.max(feature, 2)
        end_points['key_point_indices'] = max_indices
        x = self.mlp_global(x)

        cls_logit = self.classifier(x)

        preds = {
            'cls_logit': cls_logit
        }
        preds.update(end_points)

        return preds

    def reset_parameters(self):
        for sa_module in self.sa_modules:
            sa_module.reset_parameters(xavier_uniform)
        self.mlp_global.reset_parameters(xavier_uniform)
        xavier_uniform(self.classifier)
        set_bn(self, momentum=0.01)


def test_PointNet2MSGCls():
    batch_size = 8
    in_channels = 6
    num_points = 1024
    num_classes = 40

    points = torch.randn(batch_size, in_channels, num_points)
    points = points.cuda()

    pn2msg = PointNet2MSGCls(in_channels, num_classes)
    pn2msg.cuda()
    out_dict = pn2msg({'points': points})
    for k, v in out_dict.items():
        print('PointNet2MSG:', k, v.shape)
