import torch

from core.nn import SharedMLP
from .functions import get_edge_feature


class EdgeConvBlock(SharedMLP):
    """EdgeConv Block"""

    def __init__(self, in_channels, conv_channels, k, **kwargs):
        super(EdgeConvBlock, self).__init__(
            in_channels=2 * in_channels,
            mlp_channels=conv_channels,
            ndim=2,
            **kwargs
        )
        self.k = k

    def forward(self, x):
        # input x: (batch_size, channels, num_nodes)
        x = get_edge_feature(x, self.k)  # (batch_size, 2* channels, num_nodes, k)
        x = super(EdgeConvBlock, self).forward(x)  # (batch_size, conv_channels[-1], num_nodes, k)
        x, _ = torch.max(x, 3)  # (batch_size, conv_channels[-1], num_nodes)
        return x

    def extra_repr(self):
        extra_str = 'k={}'.format(self.k)
        other_str = super(EdgeConvBlock, self).extra_repr()
        return extra_str if other_str == '' else ','.join([extra_str, other_str])
