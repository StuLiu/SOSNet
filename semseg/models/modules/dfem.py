"""
@Project : semantic-segmentation 
@File    : dfem.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/6/6 上午9:53
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn as nn
import numpy as np
from semseg.models.backbones.mobilenetv3_ import InvertedResidual, _make_divisible, conv_3x3_bn
from torch.nn import init

dfem_settings = {
    'MobileNetV3': {
        'large': [[
            # k, t, c, SE, HS, s        # 0 <- conv_3x3_bn(3, input_channel, 2)
            [3, 1, 16, 0, 0, 1],  # 1
            [3, 4, 24, 0, 0, 2],  # 2
            [3, 3, 24, 0, 0, 1],  # 3
            [5, 3, 40, 1, 0, 2],  # 4
            [5, 3, 40, 1, 0, 1],  # 5
            [5, 3, 40, 1, 0, 1],  # 6
            [3, 6, 80, 0, 1, 2],  # 7
            [3, 2.5, 80, 0, 1, 1],  # 8
            [3, 2.3, 80, 0, 1, 1],  # 9
            [3, 2.3, 80, 0, 1, 1],  # 10
            [3, 6, 112, 1, 1, 1],  # 11
            [3, 6, 112, 1, 1, 1],  # 12
            [5, 6, 160, 1, 1, 2],  # 13
            [5, 6, 160, 1, 1, 1],  # 14
            [5, 6, 160, 1, 1, 1]], [0, 4, 7, 13, 16], [24, 40, 112, 160]
        ],
        'small': [[
            # k, t, c, SE, HS, s        # 0 <- conv_3x3_bn(3, input_channel, 2)
            [3, 1, 16, 1, 0, 2],  # 1
            [3, 4.5, 24, 0, 0, 2],  # 2
            [3, 3.67, 24, 0, 0, 1],  # 3
            [5, 4, 40, 1, 1, 2],  # 4
            [5, 6, 40, 1, 1, 1],  # 5
            [5, 6, 40, 1, 1, 1],  # 6
            [5, 3, 48, 1, 1, 1],  # 7
            [5, 3, 48, 1, 1, 1],  # 8
            [5, 6, 96, 1, 1, 2],  # 9
            [5, 6, 96, 1, 1, 1],  # 10
            [5, 6, 96, 1, 1, 1]], [0, 2, 4, 9, 12], [16, 24, 48, 96]
        ]
    }
}


class DetailFeatureEnhanceModuleABL(nn.Module):

    def __init__(self, backbone='MobileNetV3-large', width_mult=1.):
        super().__init__()
        backbone_base, backbone_level = backbone.split('-')
        assert backbone_base in ['MobileNetV3', 'MobileNetV2', 'ResNet', ], 'unsupported backbone'
        self.cfgs, self.divs, self.channels = dfem_settings[backbone_base][backbone_level]
        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs[: self.divs[2]]:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        outs = []
        out = x
        for i in range(0, 2):
            out = self.features[self.divs[i]:self.divs[i + 1]](out)
            outs.append(out)
        return *outs, outs[-1]

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
    model = DetailFeatureEnhanceModuleABL('MobileNetV3-large')
    model.train()
    _x = torch.randn(1, 3, 512, 512)
    _outs = model(_x)
    for _y in _outs:
        print(_y.shape)

    from semseg.utils.utils import model_summary, init_logger

    init_logger()
    model_summary(model, (1, 3, 224, 224))
