from einops import rearrange
import torch
from torch import nn

# https://arxiv.org/abs/2101.11605
class BottleneckMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, feature_map_size, heads=4, dim_head=None,
                 content_based_positional_encoding = False):
        super().__init__()
        self.heads = heads
        self.dim_head = (int(dim/heads)) if dim_head is None else dim_head
        self.scale = dim_head ** -0.5
        self.feature_map_size = feature_map_size
        self.to_qkv = nn.Conv2d(dim, heads*self.dim_head*3, 1, bias=False)
        self.height = self.feature_map_size[0]
        self.width = self.feature_map_size[1]

    def forward(self, x):
        # [batch_size (heads*dim_head*3) height width]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(qkv,
                                          'b (d k h) x y -> k b h (x y) d',
                                          k=3, h=self.heads))
        dots = torch.einsum('b h i d, b h j d -> b h i j',q, v)
        dots *= self.scale
        attention = dots.softmax(-1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=self.height,
                        y=self.width)
        return out
