'''
@Project : semantic-segmentation
@File    : mobilenetv3.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2022/5/26 下午3:08
@e-mail  : 1183862787@qq.com

Modified from https://github.com/xiaolai-sqlai/mobilenetv3.
The implementation of SeModule was wrong

MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se is not None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


mobilenetv3_settings = {
    'small': [[0, 1, 2, 4, 11], [12, 24, 40, 96]],
    'large': [[0, 2, 4, 7, 15], [24, 40, 80, 160]],
}


class MobileNetV3(nn.Module):
    def __init__(self, model_name='large'):
        super(MobileNetV3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        if model_name == 'small':
            self.bneck = nn.Sequential(
                Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
                Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
                Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
                Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
                Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
                Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
                Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
                Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
                Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
                Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
                Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            )
        elif model_name == 'large':
            self.bneck = nn.Sequential(
                Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
                Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
                Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
                Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
                Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
                Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
                Block(3, 40, 240, 80, hswish(), None, 2),
                Block(3, 80, 200, 80, hswish(), None, 1),
                Block(3, 80, 184, 80, hswish(), None, 1),
                Block(3, 80, 184, 80, hswish(), None, 1),
                Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
                Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
                Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
                Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
                Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
            )
        else:
            raise Exception('not small or large')

        self.divs, self.channels = mobilenetv3_settings[model_name]
        self.init_params()

    def init_params(self):
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

    def forward(self, x):
        outs = []
        out = self.hs1(self.bn1(self.conv1(x)))
        for i in range(0, 4):
            out = self.bneck[self.divs[i]:self.divs[i + 1]](out)
            outs.append(out)
        return outs


if __name__ == '__main__':
    model = MobileNetV3('large')
    model.load_state_dict(torch.load('../../../checkpoints/backbones/mobilenet/mobilenetv3_large.pth',
                                     map_location='cpu'), strict=False)
    model.train()
    _x = torch.randn(1, 3, 512, 512)
    _outs = model(_x)
    for y in _outs:
        print(y.shape)

    from semseg.utils.utils import model_summary, init_logger

    init_logger()
    model_summary(model, (1, 3, 224, 224))

    # from fvcore.nn import flop_count_table, FlopCountAnalysis
    # print(flop_count_table(FlopCountAnalysis(model, _x.cuda())))
