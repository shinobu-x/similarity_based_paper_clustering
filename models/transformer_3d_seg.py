from einops import rearrange
import torch
from torch import nn
from models.transformer_encoder import TransformerEncoder

# https://arxiv.org/abs/2102.13645
class Transformer3DSeg(nn.Module):
    def __init__(self, *, subvolume_dim=24, patch_dim=8, num_classes=2,
                 input_channels=3, dim=1024, depth=8, blocks=7, heads=4,
                 dim_mlp=1024, dim_head=None, dropout=0.0):
        super().__init__()
        self.patch_dim = patch_dim
        self.num_classes = num_classes
        self.n = subvolume_dim//patch_dim
        self.tokens = self.n**3
        self.mid_tokens = self.tokens//2
        self.token_dim = input_channels*(self.patch_dim**3)
        self.linear = nn.Linear(self.token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.tokens + 1, dim))
        self.dropout = nn.Dropout(dropout)
        dim_head = (int(dim/heads)) if dim_head is None else dim_head
        self.transformer = TransformerEncoder(dim, depth, blocks, heads,
                                              dim_head, dim_mlp, dropout)
        self.mlp_head = nn.Linear(dim, self.tokens*self.num_classes)

    def forward(self, x, mask=None):
        # [batch, channels, h, w, z] -> [batch, tokens, patch_volume]
        image_patches = rearrange(x,
                            'b c (patch_x x) (patch_y y) (patch_z z) -> \
                             b (x y z) (patch_x patch_y patch_z c)',
                            patch_x=self.patch_dim,
                            patch_y=self.patch_dim,
                            patch_z=self.patch_dim)
        b, tokens, _ = image_patches.shape
        image_patches = self.linear(image_patches)
        embeddings = self.dropout(image_patches)
        x = self.transformer(embeddings, mask)
        out = self.mlp_head(x[:, self.mid_tokens, :])
        out = rearrange(out, 'b (x y z c) -> b c x y z',
                      x=self.n, y=self.n, z=self.n)
        return out

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer3DSeg(num_classes=4).to(device)
    inputs = torch.rand(1, 3, 128, 128, 128).to(device)
    y = model(inputs)
    print(y.shape)
