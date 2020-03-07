import torch
import torch.nn as nn

from core.nn.init import xavier_uniform, set_bn
from core.nn import SharedMLP, MLP, Conv1d
from IPython import embed


class ConcatHead(nn.Module):

    def __init__(self, in_channels, dropout_prob=0.5):
        super(ConcatHead, self).__init__()
        self.mlp_local = SharedMLP(in_channels, (in_channels,), dropout_prob=dropout_prob)
        self.conv1d = Conv1d(in_channels, 2, 1, relu=False, bn=False)
        #self.classifier = nn.Linear(in_channels, 2, bias=True)
        #self.mlp_local = SharedMLP(in_channels, (in_channels,2), dropout_prob=dropout_prob)
        self.reset_parameters()

    def forward(self, concat_feats):
        if isinstance(concat_feats, (list, tuple)):
            # concat_feature, (batch_size, in_channel, num_points)
            concat_feats = torch.cat(concat_feats, dim=1)
        # ins_logit, (batch_size, 2, num_points)
        ins_logit = self.mlp_local(concat_feats)
        ins_logit = self.conv1d(ins_logit)

        return ins_logit

    def reset_parameters(self):
        #xavier_uniform(self.classifier)
        self.mlp_local.reset_parameters(xavier_uniform)
        self.conv1d.reset_parameters(xavier_uniform)
        set_bn(self, momentum=0.01)
