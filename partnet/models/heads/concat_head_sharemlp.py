import torch
import torch.nn as nn

from core.nn.init import xavier_uniform, set_bn
from core.nn import SharedMLP, MLP
from IPython import embed


class ConcatHead(nn.Module):

    def __init__(self, in_channels, dropout_prob=0.5):
        super(ConcatHead, self).__init__()
        #self.ins_logit = nn.Conv1d(in_channels, logit_channels, 1, bias=True)
        self.mlp_local = SharedMLP(in_channels, (in_channels,2), dropout_prob=dropout_prob)
        self.reset_parameters()

    def forward(self, concat_feats):
        if isinstance(concat_feats, (list, tuple)):
            # concat_feature, (batch_size, in_channel, num_points)
            concat_feats = torch.cat(concat_feats, dim=1)
        # ins_logit, (batch_size, 2, num_points)
        ins_logit = self.mlp_local(concat_feats)

        return ins_logit

    def reset_parameters(self):
        #xavier_uniform(self.ins_logit)
        self.mlp_local.reset_parameters(xavier_uniform)
        set_bn(self, momentum=0.01)
