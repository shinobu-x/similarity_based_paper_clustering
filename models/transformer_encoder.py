from torch import nn
from models.transformer import Transformer

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, blocks=6, heads=8, dim_head=None,
                 dim_mlp=1024, dropout = 0.0, prenorm=False):
        super().__init__()
        self.blocks = [Transformer(dim, depth, heads, dim_head, dim_mlp,
                                   dropout, prenorm=prenorm) \
                       for _ in range(blocks)]
        self.layers = nn.ModuleList(self.blocks)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
