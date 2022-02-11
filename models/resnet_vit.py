import torch
from torch import nn
from models.resnet import ResNet, Bottleneck
from models.vit import ViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ResNetViT(nn.Module):
    def __init__(self, input_channels, image_size, patch_size, num_classes,
                 dim, depth, blocks, heads, dim_mlp, use_fc, resnet_type=50):
        super().__init__()
        resnet_types = {50:[3,4,6,3],
                       101:[3,4,23,3],
                       152:[3,8,36,3]}
        resnet = ResNet(Bottleneck, resnet_types[resnet_type],
                        num_classes=num_classes,
                        num_input_channels=input_channels, use_fc=use_fc)
        resnet_layers = list(resnet.children())[:5]
        self.resnet = nn.Sequential(*resnet_layers)
        self.vit = ViT(image_size=image_size, patch_size=patch_size,
                       num_classes=num_classes, dim=dim, depth=depth,
                       blocks=blocks, heads=heads, dim_mlp=dim_mlp,
                       input_channels = input_channels+1)

    def forward(self, x):
        x =self.resnet(x)
        x = x.reshape(4, 4, 224, 224)
        x = self.vit(x)
        return x

if __name__ == '__main__':
    model = ResNetViT(input_channels=3, image_size=224, patch_size=16,
                      num_classes=10, dim=512, depth=16, blocks=6, heads=8,
                      dim_mlp=1024, use_fc=False, resnet_type=152).to(device)
    input = torch.randn(4, 3, 224, 224).to(device)
    y = model(input)
    print(y.shape)
