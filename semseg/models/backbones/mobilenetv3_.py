"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan,
Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch
import torch.nn as nn
from torch.nn import init
import math

__all__ = ['MobileNetV3', 'mobilenetv3_settings']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        """
        Basic block for constructing the MobileNetV3
        Args:
            inp: int; number of input channels
            hidden_dim: int; number of middle channels
            oup: int; number of output channels
            kernel_size: int; conv kernel size
            stride: int; conv stride size
            use_se: bool; using SELayer if True, else Identity between two convs.
            use_hs: bool; using h_swish as the activation if True, else ReLU.
        """
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


mobilenetv3_settings = {
    # 'mode':[[block_settings], [fpn_index_settings], [fpn_channel_settings]]
    'large': [[
        # k, t, c, SE, HS, s        # 0 conv_3x3_bn(3, input_channel, 2)
        [3, 1, 16, 0, 0, 1],        # 1
        [3, 4, 24, 0, 0, 2],        # 2
        [3, 3, 24, 0, 0, 1],        # 3
        [5, 3, 40, 1, 0, 2],        # 4
        [5, 3, 40, 1, 0, 1],        # 5
        [5, 3, 40, 1, 0, 1],        # 6
        [3, 6, 80, 0, 1, 2],        # 7
        [3, 2.5, 80, 0, 1, 1],      # 8
        [3, 2.3, 80, 0, 1, 1],      # 9
        [3, 2.3, 80, 0, 1, 1],      # 10
        [3, 6, 112, 1, 1, 1],       # 11
        [3, 6, 112, 1, 1, 1],       # 12
        [5, 6, 160, 1, 1, 2],       # 13
        [5, 6, 160, 1, 1, 1],       # 14
        [5, 6, 160, 1, 1, 1]], [0, 4, 7, 13, 16], [24, 40, 112, 160]
    ],
    'small': [[
        # k, t, c, SE, HS, s        # 0 conv_3x3_bn(3, input_channel, 2)
        [3, 1, 16, 1, 0, 2],        # 1
        [3, 4.5, 24, 0, 0, 2],      # 2
        [3, 3.67, 24, 0, 0, 1],     # 3
        [5, 4, 40, 1, 1, 2],        # 4
        [5, 6, 40, 1, 1, 1],        # 5
        [5, 6, 40, 1, 1, 1],        # 6
        [5, 3, 48, 1, 1, 1],        # 7
        [5, 3, 48, 1, 1, 1],        # 8
        [5, 6, 96, 1, 1, 2],        # 9
        [5, 6, 96, 1, 1, 1],        # 10
        [5, 6, 96, 1, 1, 1]], [0, 2, 4, 9, 12], [16, 24, 48, 96]
    ]
}


class MobileNetV3(nn.Module):

    def __init__(self, model_name, width_mult=1.):
        super(MobileNetV3, self).__init__()
        assert model_name in ['large', 'small']

        self.cfgs, self.divs, self.channels = mobilenetv3_settings[model_name]
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        outs = []
        out = x
        for i in range(0, 4):
            out = self.features[self.divs[i]:self.divs[i + 1]](out)
            outs.append(out)
        return outs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = MobileNetV3('large')
    model.load_state_dict(torch.load('../../../checkpoints/backbones/mobilenet_/mobilenetv3_large.pth',
                                     map_location='cpu'), strict=False)
    model.train()
    _x = torch.randn(1, 3, 512, 512)
    _outs = model(_x)
    for _y in _outs:
        print(_y.shape)

    from semseg.utils.utils import model_summary, init_logger

    init_logger()
    model_summary(model, (1, 3, 224, 224))
