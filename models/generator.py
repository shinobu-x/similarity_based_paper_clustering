from torch import nn
from models.resnet import resnet101

class Generator(nn.Module):
    def __init__(self, image_size, num_classes, input_channels,
                 output_channels):
        super().__init__()
        self.image_size = image_size
        self.conv = nn.Conv2d(input_channels, output_channels, 3)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.feature_extractor = resnet101(num_classes=num_classes,
                                           num_input_channels=output_channels,
                                           use_fc=False)


    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.feature_extractor(x)
        return x
