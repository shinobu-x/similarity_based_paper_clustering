import torch
import torch.nn as nn
from modules import Subtraction, Subtraction2, Aggregation

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)

def position(h, w):
    loc_w = torch.linspace(-1.0, 1.0, w).to(device).unsqueeze(0).repeat(h, 1)
    loc_h = torch.linspace(-1.0, 1.0, h).to(device).unsqueeze(1).repeat(1, w)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc

class SAM(nn.Module):
    def __init__(self, module_type, in_planes, rel_planes, out_planes,
                 share_planes, kernel_size=3, stride=1, dilation=1):
        super(SAM, self).__init__()
        self.module_type = module_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        loc_planes = out_planes // share_planes
        # patchwise
        if module_type == 0:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes + 2),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes + 2, rel_planes,
                                                  kernel_size=1, bias=False),
                                        nn.BatchNorm2d(rel_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes, loc_planes,
                                                  kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride,
                                           (dilation *
                                            (kernel_size - 1) + 1) // 2,
                                           dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride,
                                             (dilation *
                                              (kernel_size - 1) + 1) // 2,
                                             dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        # pairwise
        else:
            self.conv_w = nn.Sequential(nn.BatchNorm2d(rel_planes *
                                                    (pow(kernel_size, 2) + 1)),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(rel_planes *
                                                  (pow(kernel_size, 2) + 1),
                                                  loc_planes, kernel_size=1,
                                                  bias=False),
                                        nn.BatchNorm2d(loc_planes),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(loc_planes, pow(
                                            kernel_size,2) * loc_planes,
                                                  kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation,
                                      padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size,
                                      dilation=dilation, padding=0,
                                      stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.aggregation = Aggregation(kernel_size, stride,
                                       (dilation * (kernel_size - 1) + 1) // 2,
                                       dilation, pad_mode=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        # pairwise
        if self.module_type == 0:
            p = self.conv_p(position(x.shape[2], x.shape[3]))
            w = self.softmax(self.conv_w(torch.cat(
                [self.subtraction2(x1, x2),
                 self.subtraction(p).repeat(x.shape[0], 1, 1, 1)], 1)))
        # patchwise
        else:
            if self.stride != 1:
                x1 = self.unfold_i(x1)
            x1 = x1.view(x.shape[0], -1, 1, x.shape[2]*x.shape[3])
            x2 = self.unfold_j(self.pad(x2)).view(x.shape[0], -1, 1,
                                                  x1.shape[-1])
            w = self.conv_w(torch.cat([x1, x2], 1)).view(
                x.shape[0], -1, pow(self.kernel_size, 2), x1.shape[-1])
        x = self.aggregation(x3, w)
        return x

class Bottleneck(nn.Module):
    def __init__(self, module_type, in_planes, rel_planes, mid_planes,
                 out_planes, share_planes=8, kernel_size=7, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # rel_planes = in_planes // 16
        # mid_planes = in_planes // 4
        # out_planes = in_planes
        self.sam = SAM(module_type, in_planes, rel_planes, mid_planes,
                       share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.relu(self.bn2(self.sam(out)))
        out = self.conv(out)
        out += identity
        return out

class SAN(nn.Module):
    def __init__(self, input_channels, module_type, block, layers, kernels,
                 num_classes, use_fc = True):
        super(SAN, self).__init__()
        self.use_fc = use_fc
        c = 64 # planes
        self.conv_in =conv1x1(input_channels, c)
        self.bn_in = nn.BatchNorm2d(c)
        self.conv0 = conv1x1(c, c)
        self.bn0 = nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(module_type, block, c, layers[0],
                                       kernels[0]) # 64
        c *= 4
        self.conv1 = conv1x1(c // 4, c)
        self.bn1 = nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(module_type, block, c, layers[1],
                                       kernels[1]) # 64*4 256
        c *= 2
        self.conv2 = conv1x1(c // 2, c)
        self.bn2 = nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(module_type, block, c, layers[2],
                                       kernels[2]) # 64*4*2 512
        c *= 2
        self.conv3 = conv1x1(c // 2, c)
        self.bn3 = nn.BatchNorm2d(c)
        self.layer3 = self._make_layer(module_type, block, c, layers[3],
                                       kernels[3]) # 64*4*2*2 1024
        c *= 2
        self.conv4 = conv1x1(c // 2, c)
        self.bn4 = nn.BatchNorm2d(c)
        self.layer4 = self._make_layer(module_type, block, c, layers[4],
                                       kernels[4]) # 64*4*2*2*2 2048
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if use_fc:
            self.fc = nn.Linear(c, num_classes)

    def _make_layer(self, module_type, block, planes, blocks, kernel_size=7,
                    stride=1):
        layers = []
        for _ in range(0, blocks):
            layers.append(block(module_type, planes, planes // 16, planes // 4,
                                planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.relu(self.bn0(self.layer0(self.conv0(self.pool(x)))))
        x = self.relu(self.bn1(self.layer1(self.conv1(self.pool(x)))))
        x = self.relu(self.bn2(self.layer2(self.conv2(self.pool(x)))))
        x = self.relu(self.bn3(self.layer3(self.conv3(self.pool(x)))))
        x = self.relu(self.bn4(self.layer4(self.conv4(self.pool(x)))))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.use_fc:
            x = self.fc(x)
        return x

def san(input_channels, module_type, layers, kernels, num_classes):
    model = SAN(input_channels, module_type, Bottleneck, layers, kernels,
                num_classes)
    return model

if __name__ == '__main__':
    image_size = 224
    input_channels = 48
    model = san(input_channels, module_type=0, layers=(3, 4, 6, 8, 3),
              kernels=[3, 7, 7, 7, 7],
              num_classes=2).to(device).eval()
    x = torch.randn(4, input_channels, image_size, image_size).to(device)
    y = model(x)
    print(y.shape)
