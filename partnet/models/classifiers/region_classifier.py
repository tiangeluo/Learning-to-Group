import torch
import torch.nn as nn

from core.nn import SharedMLP, Conv1d
from core.nn.init import xavier_uniform, set_bn

import shaper.models.pointnet2.functions as _F
from ..backbones import build_backbone
from ..heads import build_head
from ..backbones.pn_backbone1 import PointNetCls as PointNetCls1
from IPython import embed


class RegionClassifier(nn.Module):
    """PointNet for part segmentation

     Args:
        backbone_cfg (dict): config use to build backbone
        head_cfg (dict): config use to build head

    """

    def __init__(self,
                 backbone,
                 head):
        super(RegionClassifier, self).__init__()

        self.backbone1 = PointNetCls1(in_channels=3, out_channels=32)
        self.backbone2 = build_backbone(backbone)
        self.head = build_head(head)

        self.reset_parameters()

    def forward(self, data_batch):
        # neighbour_xyz, (batch_size, 3, num_centroids, num_neighbours)
        neighbour_xyz_purity = data_batch['neighbour_xyz_purity']
        batch_size, _, num_centroids, num_neighbours = neighbour_xyz_purity.size()
        neighbour_xyz_purity = neighbour_xyz_purity.transpose(1, 2).contiguous().view(batch_size*num_centroids, 3, num_neighbours)
        preds = self.backbone1(neighbour_xyz_purity)
        center_feature = preds['cls_logit']
        #center_feature = (self.backbone2(neighbour_xyz_purity)).pop('feature')
        #center_feature,_ = torch.max(center_feature, -1)

        node_logit = preds['node_logit']
        node_loss = nn.CrossEntropyLoss()(node_logit, data_batch['valid_center_mask'].view(-1).long())
        _, node_pred = torch.max(node_logit,1)
        node_acc_arr = (node_pred.float()==data_batch['valid_center_mask'].view(-1).float()).float()
        node_acc = torch.mean(node_acc_arr)
        node_pos_acc = torch.mean(torch.index_select(node_acc_arr, dim=0, index=(data_batch['valid_center_mask'].view(-1)==1).nonzero().squeeze()))
        if torch.sum(data_batch['valid_center_mask']==0) ==0:
            node_neg_acc=0
        else:
            node_neg_acc = torch.mean(torch.index_select(node_acc_arr, dim=0, index=(data_batch['valid_center_mask'].view(-1)==0).nonzero().squeeze()))

        neighbour_xyz = data_batch['neighbour_xyz']
        batch_size, _, num_centroids, num_neighbours = neighbour_xyz.size()
        neighbour_xyz = neighbour_xyz.transpose(1, 2).contiguous().view(batch_size*num_centroids, 3, num_neighbours)
        backbone_output = self.backbone2(neighbour_xyz)
        neighbour_feature = backbone_output.pop('feature')
        feature_list = list()
        feature_list.append(neighbour_feature)
        feature_list.append(center_feature.unsqueeze(-1).expand_as(neighbour_feature))
        ins_logit = self.head(feature_list)
        ins_logit = ins_logit.view(batch_size, num_centroids, ins_logit.size(1), num_neighbours).transpose(1, 2).contiguous()

        ## neighbour_index, (batch_size, num_centroids, num_neighbours), input space
        #neighbour_index = data_batch['neighbour_index']
        ## centroid_xyz, (batch_size, 3, num_centroids)
        #centroid_xyz = data_batch['centroid_xyz']
        ## centroid_index, (batch_size, num_centroids), input space
        #centroid_index = data_batch['centroid_index']

        # translation normalization, done in collate
        # neighbour_xyz -= centroid_xyz.unsqueeze(-1)
        # centroid_xyz = centroid_xyz - centroid_xyz

        # TODO only use first one for NOW
        #batch_size, num_centroids, num_neighbours = neighbour_index.size()
        #neighbour_points = neighbour_xyz.transpose(1, 2).contiguous().view(batch_size*num_centroids, 3, num_neighbours)

        #backbone_output = self.extract_feature(neighbour_points)
        ## neighbour_feature, (batch_size * num_centroids, seg_features, num_neighbours)
        #neighbour_feature = backbone_output.pop('feature')
        #feature_list = list()
        #feature_list.append(neighbour_feature)

        ## neighbour_centroid_index, (batch_size * num_centroid, 1)
        #neighbour_centroid_index = data_batch['neighbour_centroid_index']
        #neighbour_centroid_index = neighbour_centroid_index.view(batch_size * num_centroids, 1)

        ## centroid_feature, (batch_size * num_centroid, seg_features, 1)
        #centroid_feature = _F.select_points(neighbour_feature, neighbour_centroid_index)
        #feature_list.append(centroid_feature.expand_as(neighbour_feature))

        #ins_logit = self.head(feature_list)
        ## ins_logit, (batch_size, 2, num_centroid, num_neighbours)
        #ins_logit = ins_logit.view(batch_size, num_centroids, ins_logit.size(1), num_neighbours).transpose(1, 2).contiguous()

        preds = {
            'ins_logit': ins_logit,
            'node_acc': node_acc,
            'node_pos_acc': node_pos_acc,
            'node_neg_acc': node_neg_acc,
            'center_valid_ratio': torch.mean(torch.sum(data_batch['valid_center_mask'],1).float()/num_centroids),
            'node_loss':node_loss,
        }

        return preds

    def reset_parameters(self):
        # default initialization
        self.backbone1.reset_parameters()
        self.backbone2.reset_parameters()
        self.head.reset_parameters()
        # set batch normalization to 0.01 as default
        set_bn(self, momentum=0.01)
