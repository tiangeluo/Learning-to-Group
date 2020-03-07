"""DGCNN

References:
    @article{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds},
      author={Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon},
      journal={arXiv preprint arXiv:1801.07829},
      year={2018}
    }
"""

import torch
import torch.nn as nn

from core.nn import MLP, SharedMLP
from core.nn.init import xavier_uniform, set_bn
from shaper.models.dgcnn.modules import EdgeConvBlock


class TNet(nn.Module):
    """Transformation Network for DGCNN

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        conv_channels (tuple of int): the numbers of channels of edge convolution layers
        local_channels (tuple of int): the numbers of channels in local mlp
        global_channels (tuple of int): the numbers of channels in global mlp
        k: the number of neareast neighbours for edge feature extractor

    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 conv_channels=(64, 128),
                 local_channels=(1024,),
                 global_channels=(512, 256),
                 k=20):
        super(TNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_conv = EdgeConvBlock(in_channels, conv_channels, k)
        self.mlp_local = SharedMLP(conv_channels[-1], local_channels)
        self.mlp_global = MLP(local_channels[-1], global_channels)
        self.linear = nn.Linear(global_channels[-1], self.in_channels * self.out_channels, bias=True)

        self.reset_parameters()

    def forward(self, x):
        # input x: (batch_size, in_channels, num_points)
        x = self.edge_conv(x)  # (batch_size, edge_channels[-1], num_points)
        x = self.mlp_local(x)  # (batch_size, local_channels[-1], num_points)
        x, _ = torch.max(x, 2)
        x = self.mlp_global(x)
        x = self.linear(x)
        x = x.view(-1, self.out_channels, self.in_channels)
        I = torch.eye(self.out_channels, self.in_channels, device=x.device)
        x = x.add(I)  # broadcast first dimension
        return x

    def reset_parameters(self, init_fn=xavier_uniform):
        self.edge_conv.reset_parameters(init_fn)
        self.mlp_local.reset_parameters(init_fn)
        self.mlp_global.reset_parameters(init_fn)
        # set linear transform be 0
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


# -----------------------------------------------------------------------------
# DGCNN for classification
# -----------------------------------------------------------------------------
class DGCNNCls(nn.Module):
    """DGCNN for classification

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        edge_conv_channels (tuple of int): the numbers of channels of edge convolution layers
        inter_channels (int): the number of channels of intermediate features
        global_channels (tuple of int): the numbers of channels in global mlp
        k (int): the number of neareast neighbours for edge feature extractor
        dropout_prob (float): the probability to dropout
        with_transform (bool): whether to use TNet to transform features.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_conv_channels=(64, 64, 64, 128),
                 local_channels=(1024,),
                 global_channels=(512, 256),
                 k=20,
                 dropout_prob=0.5,
                 with_transform=True):
        super(DGCNNCls, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_transform = with_transform

        # input transform
        if self.with_transform:
            self.transform_input = TNet(in_channels, in_channels, k=k)

        self.edge_convs = nn.ModuleList()
        inter_channels = []
        for conv_channels in edge_conv_channels:
            if isinstance(conv_channels, int):
                conv_channels = [conv_channels]
            else:
                assert isinstance(conv_channels, (tuple, list))
            self.edge_convs.append(EdgeConvBlock(in_channels, conv_channels, k))
            inter_channels.append(conv_channels[-1])
            in_channels = conv_channels[-1]
        self.mlp_local = SharedMLP(sum(inter_channels), local_channels)
        self.mlp_global = MLP(local_channels[-1], global_channels, dropout_prob=dropout_prob)
        self.classifier = nn.Linear(global_channels[-1], self.out_channels, bias=True)

        self.reset_parameters()

    def forward(self, data_batch):
        x = data_batch['points']
        end_points = {}

        # input transform
        if self.with_transform:
            trans_input = self.transform_input(x)
            x = torch.bmm(trans_input, x)
            end_points['trans_input'] = trans_input

        # EdgeConv
        features = []
        for edge_conv in self.edge_convs:
            x = edge_conv(x)
            features.append(x)

        x = torch.cat(features, dim=1)

        x = self.mlp_local(x)
        x, max_indices = torch.max(x, 2)
        end_points['key_point_indices'] = max_indices
        x = self.mlp_global(x)
        x = self.classifier(x)
        preds = {
            'cls_logit': x,
        }
        preds.update(end_points)

        return preds

    def reset_parameters(self):
        for edge_conv in self.edge_convs:
            edge_conv.reset_parameters(xavier_uniform)
        self.mlp_local.reset_parameters(xavier_uniform)
        self.mlp_global.reset_parameters(xavier_uniform)
        xavier_uniform(self.classifier)
        set_bn(self, momentum=0.01)


def test_DGCNNCls():
    batch_size = 4
    in_channels = 3
    num_points = 1024
    num_classes = 40

    points = torch.randn(batch_size, in_channels, num_points)
    points = points.cuda()

    tnet = TNet()
    tnet = tnet.cuda()
    out = tnet(points)
    print('TNet: ', out.size())

    dgcnn = DGCNNCls(in_channels, num_classes)
    dgcnn = dgcnn.cuda()
    out_dict = dgcnn({'points': points})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)
