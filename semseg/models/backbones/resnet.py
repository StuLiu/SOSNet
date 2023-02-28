import torch
from torch import nn, Tensor
from torch.nn import functional as F


class BasicBlock(nn.Module):
    """2 Layer No Expansion Block
    """
    expansion: int = 1

    def __init__(self, c1, c2, s=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class Bottleneck(nn.Module):
    """3 Layer 4x Expansion Block
    """
    expansion: int = 4

    def __init__(self, c1, c2, s=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


resnet_settings = {
    '18': [BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512]],
    '34': [BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512]],
    '50': [Bottleneck, [3, 4, 6, 3], [256, 512, 1024, 2048]],
    '101': [Bottleneck, [3, 4, 23, 3], [256, 512, 1024, 2048]],
    '152': [Bottleneck, [3, 8, 36, 3], [256, 512, 1024, 2048]]
}


class ResNet(nn.Module):
    def __init__(self, model_name: str = '50') -> None:
        super().__init__()
        assert model_name in resnet_settings.keys(), f"ResNet model name should be in {list(resnet_settings.keys())}"
        self.block, self.depths, self.channels = resnet_settings[model_name]

        self.inplanes = 64  # the input channel count, changed after _make_layer to apply to next block.
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(self.block, 64, self.depths[0], s=1)
        self.layer2 = self._make_layer(self.block, 128, self.depths[1], s=2)
        self.layer3 = self._make_layer(self.block, 256, self.depths[2], s=2)
        self.layer4 = self._make_layer(self.block, 512, self.depths[3], s=2)

    def _make_layer(self, block, planes, depth, s=1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = nn.Sequential(
            block(self.inplanes, planes, s, downsample),
            *[block(planes * block.expansion, planes) for _ in range(1, depth)]
        )
        self.inplanes = planes * block.expansion
        return layers

    def forward(self, x: Tensor):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))  # [1, 64, H/4, W/4]
        x1 = self.layer1(x)  # [1, 64/256, H/4, W/4]   
        x2 = self.layer2(x1)  # [1, 128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [1, 256/1024, H/16, W/16]
        x4 = self.layer4(x3)  # [1, 512/2048, H/32, W/32]
        return x1, x2, x3, x4


if __name__ == '__main__':
    _model = ResNet('101')
    # model.load_state_dict(torch.load('C:\\Users\\sithu\\Documents\\weights\\backbones\\resnet\\resnet18_a1.pth',
    # map_location='cpu'), strict=False)
    _x = torch.zeros(1, 3, 512, 512)
    _outs = _model(_x)
    for _y in _outs:
        print(_y.shape)
    from semseg.utils.utils import count_parameters

    print(f'model params cnt: {count_parameters(_model)}MB')
