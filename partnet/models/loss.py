import torch
from torch import nn as nn
import torch.nn.functional as F

from core.nn.functional import cross_entropy
from core.nn.functional import focal_loss
from core.nn.functional import binary_cross_entropy_with_logit
from core.nn.functional import smooth_cross_entropy


class PairInsSegLoss(nn.Module):
    """Pointnet part segmentation loss with optional regularization loss"""

    def __init__(self, ins_weight=1., sampler=None):
        super(PairInsSegLoss, self).__init__()
        assert ins_weight > 0.0
        self.ins_weight = ins_weight
        self.sampler = sampler

    def forward(self, preds, data_batch):
        # ins_logit, (batch_size * num_centroid, 2, num_neighbours)
        ins_logit = preds['ins_logit']
        if 'ins_label' not in data_batch.keys():
            # neighbour_index, (batch_size, num_centroids, num_neighbours)
            neighbour_index = preds['neighbour_index']
            # centroid_index, (batch_size, num_centroids)
            centroid_index = preds['centroid_index']
            # ins_id, (batch_size, length)
            ins_id = data_batch['ins_id']
            batch_size, num_centroids, num_neighbours = neighbour_index.size()
            # neighbour_label, (batch_size, num_centroids, num_neighbours)
            neighbour_label = ins_id.gather(1, neighbour_index.view(batch_size, num_centroids*num_neighbours))
            neighbour_label = neighbour_label.view(batch_size, num_centroids, num_neighbours)
            # centroid_label, (batch_size, num_centroids, 1)
            centroid_label = ins_id.gather(1, centroid_index).view(batch_size, num_centroids, 1)
            # ins_label, (batch_size, num_centroids, num_neighbours)
            ins_label = (neighbour_label == centroid_label.expand_as(neighbour_label)).long()

            # TODO add ins_label to data_batch hack
            data_batch['ins_label'] = ins_label
        else:
            ins_label = data_batch['ins_label']

        loss_fn = cross_entropy
        #loss_fn = focal_loss
        valid_mask = data_batch['valid_mask']
        if self.sampler is not None:
            weight = self.sampler(ins_logit, ins_label, loss_fn) * valid_mask.float()
        else:
            weight = valid_mask.float()
        ins_loss = loss_fn(ins_logit, ins_label, weight)
        loss_dict = {
            'ins_loss': ins_loss * self.ins_weight,
            #'node_loss': 1/10*preds['node_loss'],
        }

        return loss_dict
