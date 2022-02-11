import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.transformer_encoder import TransformerEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 blocks, heads, dim_mlp, pool = 'cls', input_channels = 3,
                 dim_head = None, dropout = 0., emb_dropout = 0.,
                 use_fc = True):
        super().__init__()
        self.patch_size = patch_size
        self.use_fc = use_fc
        image_height, image_width = (image_size, image_size)
        patch_height, patch_width = (patch_size, patch_size)
        assert image_height % patch_height == 0 and \
            image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * \
                      (image_width // patch_width)
        patch_dim = input_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, \
            'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        dim_head = (int(dim/heads)) if dim_head is None else dim_head
        self.transformer = TransformerEncoder(dim, depth, blocks, heads,
                                              dim_head, dim_mlp, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        if use_fc:
            self.mlp_head = nn.Sequential(nn.LayerNorm(dim),
                                          nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, token, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(token + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        if self.use_fc:
            x = self.mlp_head(x)
        return x

if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import functional, Compose, Resize
    image_path = '/home/kinjo/samples/320px-Sloth_in_a_tree_(Unsplash).jpg'
    image = Image.open(image_path)
    image = functional.to_tensor(Compose([Resize(
        (224,224))])(image)).reshape(1,3,224,224).to(device)
    image_size = 224
    input_channels = image.size()[1]
    model = ViT(image_size = image_size, patch_size = 16, num_classes = 10,
                dim = 512, depth = 16, blocks = 6, heads = 8, dim_mlp = 1024,
                input_channels = input_channels).to(device)
    x = torch.randn(4, input_channels, image_size, image_size).to(device)
    y = model(image)
