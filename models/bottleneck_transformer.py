from torch import nn
from bottleneck_multi_head_self_attention \
    import BottleneckMultiHeadSelfAttention

# https://arxiv.org/abs/2101.11605
class BottleneckTransformer(nn.Module):
    def __init__(self, *, input_channels, feature_map_size, output_channels,
                 projection_factor=4, heads=4, dim_head=None, pooling=False,
                 content_based_positional_encoding=False):
        super().__init__()
