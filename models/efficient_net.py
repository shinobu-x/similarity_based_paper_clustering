import torch
from torch import nn
from torch.nn import functional as F
from misc import round_filters, round_repeats, drop_connect
from misc import get_same_padding_conv2d, efficientnet_params, get_model_params
from misc import Swish, MemoryEfficientSwish
from misc import calculate_output_image_size

'''
'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
'efficientnet-b8', 'efficientnet-l2'
'''
class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and \
                      (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup,
                                       kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom,
                                       eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup,
                                      kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom,
                                   eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = \
                max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup,
                                     out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels,
                                     out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup,
                                    kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom,
                                   eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)
        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x
        x = self._project_conv(x)
        x = self._bn2(x)
        input_filters = self._block_args.input_filters
        output_filters = self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and \
           input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        in_channels = 143
        out_channels = round_filters(32, self._global_params)
        self._conv_stem_47 = Conv2d(47, out_channels, kernel_size=3, stride=2,
                                    bias=False)
        self._conv_stem_48 = Conv2d(48, out_channels, kernel_size=3, stride=2,
                                    bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom,
                                   eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params),
                num_repeat=round_repeats(block_args.num_repeat,
                                         self._global_params))
            self._blocks.append(MBConvBlock(block_args, self._global_params,
                                            image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:
                block_args = \
                        block_args._replace(input_filters=block_args.output_filters,
                                            stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params,
                                                image_size=image_size))
        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom,
                                   eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        endpoints = dict()
        print(inputs.size()[1])
        if inputs.size()[1] == 47:
            x = self._swish(self._bn0(self._conv_stem_47(inputs)))
        elif inputs.size()[1] == 48:
            x = self._swish(self._bn0(self._conv_stem_48(inputs)))
        prev_x = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        return endpoints

    def extract_features(self, inputs):
        if inputs.size()[1] == 47:
            x = self._swish(self._bn0(self._conv_stem_47(inputs)))
        elif inputs.size()[1] == 48:
            x = self._swish(self._bn0(self._conv_stem_48(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self,x,inference=False):
        if not inference:
            x_continue = x[:,0:47,:,]
            x_location = x[:,47:95,:,]
            x_rotation = x[:,95:143,:,]
            x_continue = self._avg_pooling(self.extract_features(x_continue))
            x_location = self._avg_pooling(self.extract_features(x_location))
            x_rotation = self._avg_pooling(self.extract_features(x_rotation))
            x_continue = x_continue.view(x_continue.size(0),-1)
            x_location = x_location.view(x_location.size(0),-1)
            x_rotation = x_rotation.view(x_rotation.size(0),-1)
            return x_continue, x_location, x_rotation
        else:
            x = self._avg_pooling(self.extract_features(x))
            x = x.view(x.size(0),-1)
            return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        _, _, res, _ = efficientnet_params(model_name)
        return res

    def _change_in_channels(self, in_channels):
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3,
                                     stride=2, bias=False)
