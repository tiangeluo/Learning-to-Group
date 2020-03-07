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

from core.nn import SharedMLP, Conv1d
from core.nn.init import set_bn, xavier_uniform
from shaper.models.dgcnn.dgcnn_cls import TNet
from shaper.models.dgcnn.modules import EdgeConvBlock


# -----------------------------------------------------------------------------
# DGCNN for part segmentation
# -----------------------------------------------------------------------------
class DGCNNPartSeg(nn.Module):
    """DGCNN for part segmentation

    Args:
        in_channels (int): the number of input channels
        num_classes (int): the number of classification class
        num_seg_classes (int): the number of segmentation class
        edge_conv_channels (tuple of int): the numbers of channels of edge convolution layers
        local_channels (tuple of int): the number of channels of intermediate features
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        k (int): the number of neareast neighbours for edge feature extractor
        dropout_prob (float): the probability to dropout
        with_transform (bool): whether to use TNet to transform features.

    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_seg_classes,
                 edge_conv_channels=((64, 64), (64, 64), 64),
                 local_channels=(1024,),
                 seg_channels=(256, 256, 128),
                 k=20,
                 dropout_prob=0.4,
                 with_transform=True):
        super(DGCNNPartSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
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

        LABEL_CHANNELS = 64
        self.mlp_label = Conv1d(self.num_classes, LABEL_CHANNELS, 1)
        self.mlp_local = SharedMLP(sum(inter_channels), local_channels)

        mlp_seg_in_channels = sum(inter_channels) + local_channels[-1] + LABEL_CHANNELS
        self.mlp_seg = SharedMLP(mlp_seg_in_channels, seg_channels[:-1], dropout_prob=dropout_prob)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1)
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        self.reset_parameters()

    def forward(self, data_batch):
        x = data_batch['points']
        num_points = x.shape[2]
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

        inter_feature = torch.cat(features, dim=1)  # (batch_size, sum(inter_channels), num_points)
        x = self.mlp_local(inter_feature)
        global_feature, max_indices = torch.max(x, 2)  # (batch_size, local_channels[-1])
        # end_points['key_point_indices'] = max_indices
        global_feature_expand = global_feature.unsqueeze(2).expand(-1, -1, num_points)

        with torch.no_grad():
            cls_label = data_batch['cls_label']
            one_hot = cls_label.new_zeros(cls_label.size(0), self.num_classes)
            one_hot = one_hot.scatter(1, cls_label.unsqueeze(1), 1).float()  # (batch_size, num_classes)
            one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points)
        label_feature = self.mlp_label(one_hot_expand)

        # (batch_size, mlp_seg_in_channels, num_points)
        x = torch.cat((inter_feature, global_feature_expand, label_feature), dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        seg_logit = self.seg_logit(x)
        preds = {
            'seg_logit': seg_logit,
        }
        preds.update(end_points)

        return preds

    def reset_parameters(self):
        for edge_conv in self.edge_convs:
            edge_conv.reset_parameters(xavier_uniform)
        self.mlp_label.reset_parameters(xavier_uniform)
        self.mlp_local.reset_parameters(xavier_uniform)
        self.mlp_seg.reset_parameters(xavier_uniform)
        self.conv_seg.reset_parameters(xavier_uniform)
        xavier_uniform(self.seg_logit)
        set_bn(self, momentum=0.01)


def test_DGCNNPartSeg():
    batch_size = 8
    in_channels = 3
    num_points = 2048
    num_classes = 16
    num_seg_classes = 50

    points = torch.rand(batch_size, in_channels, num_points)
    points = points.cuda()
    cls_label = torch.randint(num_classes, (batch_size,))
    cls_label = cls_label.cuda()

    dgcnn = DGCNNPartSeg(in_channels, num_classes, num_seg_classes)
    dgcnn = dgcnn.cuda()
    out_dict = dgcnn({'points': points, 'cls_label': cls_label})
    for k, v in out_dict.items():
        print('DGCNN:', k, v.shape)
