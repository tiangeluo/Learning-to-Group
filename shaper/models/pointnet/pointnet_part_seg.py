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


class PointNetPartSeg(nn.Module):
    """PointNet for part segmentation

     Args:
        in_channels (int): the number of input channels
        num_classes (int): the number of classification class
        num_seg_classes (int): the number of segmentation class
        stem_channels (tuple of int): the numbers of channels in stem feature extractor
        local_channels (tuple of int): the numbers of channels in local mlp
        cls_channels (tuple of int): the numbers of channels in classification mlp
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob_cls (float): the probability to dropout in classification mlp
        dropout_prob_seg (float): the probability to dropout in segmentation mlp
        with_transform (bool): whether to use TNet to transform features.
        use_one_hot (bool): whehter to use one hot vector of class labels.

    References:
        https://github.com/charlesq34/pointnet/blob/master/part_seg/pointnet_part_seg.py

    """

    def __init__(self,
                 in_channels,
                 num_classes,
                 num_seg_classes,
                 stem_channels=(64, 128, 128),
                 local_channels=(512, 2048),
                 cls_channels=(256, 256),
                 seg_channels=(256, 256, 128),
                 dropout_prob_cls=0.3,
                 dropout_prob_seg=0.2,
                 with_transform=True,
                 use_one_hot=True):
        super(PointNetPartSeg, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.use_one_hot = use_one_hot

        # stem
        self.stem = Stem(in_channels, stem_channels, with_transform=with_transform)
        self.mlp_local = SharedMLP(stem_channels[-1], local_channels)

        # part segmentation
        # Notice that the original repo concatenates global feature, one hot class embedding,
        # stem features and local features. However, the paper does not use last local feature.
        # Here, we follow the released repo.
        in_channels_seg = sum(stem_channels) + sum(local_channels) + local_channels[-1]
        if self.use_one_hot:
            in_channels_seg += num_classes
        self.mlp_seg = SharedMLP(in_channels_seg, seg_channels[:-1], dropout_prob=dropout_prob_seg)
        self.conv_seg = Conv1d(seg_channels[-2], seg_channels[-1], 1)
        self.seg_logit = nn.Conv1d(seg_channels[-1], num_seg_classes, 1, bias=True)

        # classification (optional)
        if len(cls_channels) > 0:
            # Notice that we apply dropout to each classification mlp.
            self.mlp_cls = MLP(local_channels[-1], cls_channels, dropout_prob=dropout_prob_cls)
            self.cls_logit = nn.Linear(cls_channels[-1], num_classes, bias=True)
        else:
            self.mlp_cls = None
            self.cls_logit = None

        self.reset_parameters()

    def forward(self, data_batch):
        x = data_batch['points']
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
        if self.use_one_hot:
            with torch.no_grad():
                cls_label = data_batch['cls_label']
                one_hot = cls_label.new_zeros(cls_label.size(0), self.num_classes)
                one_hot = one_hot.scatter(1, cls_label.unsqueeze(1), 1).float()  # (batch_size, num_classes)
                one_hot_expand = one_hot.unsqueeze(2).expand(-1, -1, num_points)
            seg_features.append(one_hot_expand)

        x = torch.cat(seg_features, dim=1)
        x = self.mlp_seg(x)
        x = self.conv_seg(x)
        seg_logit = self.seg_logit(x)

        preds = {
            'seg_logit': seg_logit
        }
        preds.update(end_points)

        # classification (optional)
        if self.cls_logit is not None:
            x = self.mlp_cls(global_feature)
            cls_logit = self.cls_logit(x)
            preds['cls_logit'] = cls_logit

        return preds

    def reset_parameters(self):
        # default initialization
        self.mlp_local.reset_parameters(xavier_uniform)
        self.mlp_seg.reset_parameters(xavier_uniform)
        self.conv_seg.reset_parameters(xavier_uniform)
        if self.cls_logit is not None:
            self.mlp_cls.reset_parameters(xavier_uniform)
            xavier_uniform(self.cls_logit)
        xavier_uniform(self.seg_logit)
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)


class PointNetPartSegLoss(nn.Module):
    """Pointnet part segmentation loss with optional regularization loss"""

    def __init__(self, reg_weight, cls_weight, seg_weight):
        super(PointNetPartSegLoss, self).__init__()
        assert reg_weight >= 0.0 and seg_weight > 0.0 and cls_weight >= 0.0
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight

    def forward(self, preds, labels):
        seg_logit = preds['seg_logit']
        seg_label = labels['seg_label']
        seg_loss = F.cross_entropy(seg_logit, seg_label)
        loss_dict = {
            'seg_loss': seg_loss * self.seg_weight,
        }

        if self.cls_weight > 0.0:
            cls_logit = preds['cls_logit']
            cls_label = labels['cls_label']
            cls_loss = F.cross_entropy(cls_logit, cls_label)
            loss_dict['cls_loss'] = cls_loss

        # regularization over transform matrix
        if self.reg_weight > 0.0:
            trans_feature = preds['trans_feature']
            trans_norm = torch.bmm(trans_feature.transpose(2, 1), trans_feature)  # [in, in]
            I = torch.eye(trans_norm.size(2), dtype=trans_norm.dtype, device=trans_norm.device)
            reg_loss = F.mse_loss(trans_norm, I.unsqueeze(0).expand_as(trans_norm), reduction='sum')
            loss_dict['reg_loss'] = reg_loss * (0.5 * self.reg_weight / trans_norm.size(0))

        return loss_dict


def test_PointNetPartSeg():
    batch_size = 32
    in_channels = 3
    num_points = 2048
    num_classes = 16
    num_seg_classes = 50

    points = torch.randn(batch_size, in_channels, num_points)
    cls_label = torch.randint(num_classes, (batch_size,))

    pointnet = PointNetPartSeg(in_channels, num_classes, num_seg_classes)
    out_dict = pointnet({'points': points, 'cls_label': cls_label})
    for k, v in out_dict.items():
        print('PointNet:', k, v.shape)
