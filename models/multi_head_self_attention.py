import numpy as np
import torch
from torch import nn
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = None, dropout = 0.0):
        super().__init__()
        dim_head = (int(dim/heads)) if dim_head is None else dim_head
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)) \
                                    if project_out else nn.Identity()

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                          h = self.heads), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        dots *= self.scale
        #mask = torch.rand(dots.size()[2], dots.size()[3]).bool().to('cuda')
        if mask is not None:
            assert mask.shape == dots.shape[2:]
            dots = dots.masked_fill(mask, -np.inf)
        attention = dots.softmax(-1)
        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
