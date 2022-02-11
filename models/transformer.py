from torch import nn
import torch.nn.functional as F
from models.multi_head_self_attention import MultiHeadSelfAttention

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.feed_forward = nn.Sequential(nn.Linear(dim, hidden_dim),
                                          GELU(),
                                          nn.Dropout(dropout),
                                          nn.Linear(hidden_dim, dim),
                                          nn.Dropout(dropout))

    def forward(self, x):
        return self.feed_forward(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dim_mlp, dropout = 0.0,
                 prenorm=False):
        super().__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.multi_head_self_attention = MultiHeadSelfAttention(dim,
                            heads=heads, dim_head=dim_head, dropout=dropout)
        self.feed_forward = FeedForward(dim, dim_mlp, dropout)

    def forward(self, x, mask=None):
        if self.prenorm:
            x = self.dropout(self.multi_head_self_attention(
                self.norm(x),mask)) + x
            x = self.feed_forward(self.norm(x)) + x
        else:
            x = self.norm(self.dropout(
                self.multi_head_self_attention(x, mask))+x)
            x = self.norm(self.feed_forward(x)+x)
        return x
