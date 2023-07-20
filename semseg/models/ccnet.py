import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
import functools
import time
import sys
import os
from inplace_abn import InPlaceABN, InPlaceABNSync

from semseg.models.modules.cc_attention import CrissCrossAttention
from semseg.models.backbones.resnetd import Stem, BasicBlock, Bottleneck, resnetd_settings
from semseg.models.base import BaseModel
from semseg.models.heads.upernet import UPerHead


BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
affine_par = True


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class CCNet(BaseModel):

    def __init__(self, backbone: str = 'ResNetD-18', class_num=19, recurrence=2) -> None:
        super().__init__(backbone=None, num_classes=class_num)

        variant = '101'
        if backbone is not None:
            _, variant = backbone.split('-')

        assert variant in resnetd_settings.keys(), f"ResNetD model name should be in {list(resnetd_settings.keys())}"
        self.block, self.depths, self.channels = resnetd_settings[variant]

        self.inplanes = 128
        self.stem = Stem(3, 64, self.inplanes)
        self.layer1 = self._make_layer(self.block, 64, self.depths[0], s=1)
        self.layer2 = self._make_layer(self.block, 128, self.depths[1], s=2)
        self.layer3 = self._make_layer(self.block, 256, self.depths[2], s=1, d=2)
        self.layer4 = self._make_layer(self.block, 512, self.depths[3], s=1, d=4)

        self.head_bottom = RCCAModule(self.channels[-1], 512, class_num)
        self.head_top = UPerHead(in_channels=self.channels, channel=32, num_classes=2)
        self.recurrence = recurrence

    def _make_layer(self, block, planes, depth, s=1, d=1) -> nn.Sequential:
        downsample = None

        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = nn.Sequential(
            block(self.inplanes, planes, s, d, downsample=downsample),
            *[block(planes * block.expansion, planes, d=d) for _ in range(1, depth)]
        )
        self.inplanes = planes * block.expansion
        return layers

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[-2:]
        x = self.stem(x)   # [1, 128, H/4, W/4]
        x1 = self.layer1(x)  # [1, 64/256, H/4, W/4]
        x2 = self.layer2(x1)  # [1, 128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [1, 256/1024, H/8, W/8]
        x4 = self.layer4(x3)  # [1, 512/2048, H/8, W/8]

        logits_bottom = self.head_bottom(x4, self.recurrence)
        logits_bottom = F.interpolate(logits_bottom, input_size, mode='bilinear', align_corners=True)
        if self.training:
            logits_top = self.head_top([x1, x2, x3, x4])
            logits_top = F.interpolate(logits_top, input_size, mode='bilinear', align_corners=True)
            return logits_bottom, logits_top, None
        return logits_bottom


if __name__ == '__main__':
    model = CCNet('ResNetD-18', class_num=8)
    model.init_pretrained('../../checkpoints/backbones/resnetd/resnetd18.pth')
    model.train(True).cuda()
    x = torch.zeros(32, 3, 360, 480).cuda()
    y = model(x)
    # print(y.shape)
    if model.training:
        print(y[0].shape, y[1].shape)
        pass
    else:
        print(y.shape)

