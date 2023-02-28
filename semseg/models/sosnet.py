"""
@Project : semantic-segmentation 
@File    : sosnet.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/5/26 下午3:08
@e-mail  : 1183862787@qq.com
"""
import torch
from torch import nn
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads.fcn import FCNHead
from semseg.models.heads.upernet import UPerHead
from semseg.models.modules.attention import SpatialAttention, ChannelAttention, SEModule


def conv_3x3_bn(c_in, c_out, stride, groups=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride, 1, groups=groups, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(c_in, c_out, groups=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 1, 1, 0, groups=groups, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):

    def __init__(self, c_in, c_out, c_mid):
        super().__init__()
        # pw-linear
        self.layers_begin = conv_1x1_bn(c_in=c_in, c_out=c_mid)
        # dw convs
        self.layers_mid_1 = conv_3x3_bn(c_in=c_mid, c_out=c_mid, stride=1, groups=c_mid)
        self.layers_mid_2 = conv_1x1_bn(c_in=c_mid, c_out=c_mid, groups=c_mid)
        self.layers_end = nn.Sequential(
            SEModule(c_mid + c_mid),
            nn.ReLU(inplace=True),
            # pw-linear
            conv_1x1_bn(c_in=c_mid + c_mid, c_out=c_out)
        )
        self.shortcut = nn.Sequential()
        if c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layers_begin(x)
        x2_1 = self.layers_mid_1(x1)
        x2_2 = self.layers_mid_2(x1)
        x2 = torch.cat([x2_1, x2_2], dim=1)
        x3 = self.layers_end(x2)
        return x + self.shortcut(x3)


class BasicBlock333(nn.Module):

    def __init__(self, c_in, c_out, c_mid):
        super().__init__()
        # pw-linear
        self.layers_begin = conv_1x1_bn(c_in=c_in, c_out=c_mid)
        # dw convs
        self.layers_mid_1 = conv_3x3_bn(c_in=c_mid, c_out=c_mid, stride=1, groups=c_mid)
        self.layers_end = nn.Sequential(
            SEModule(c_mid),
            nn.ReLU(inplace=True),
            # pw-linear
            conv_1x1_bn(c_in=c_mid, c_out=c_out)
        )
        self.shortcut = nn.Sequential()
        if c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.layers_begin(x)
        x2_1 = self.layers_mid_1(x1)
        x3 = self.layers_end(x2_1)
        return x + self.shortcut(x3)


class SmallObjectActivation(nn.Module):

    def __init__(self, c_in, backbone_channels):
        super().__init__()
        self.channels = backbone_channels
        # conv_ratio_size[_index]
        self.convs_1_4 = nn.Sequential(
            conv_3x3_bn(c_in=c_in, c_out=16, stride=2),
            # BasicBlock(c_in=16, c_out=16, c_mid=24),
            conv_3x3_bn(c_in=16, c_out=backbone_channels[0], stride=2),
            BasicBlock(c_in=backbone_channels[0], c_out=backbone_channels[0], c_mid=64),
        )
        self.convs_4_8 = nn.Sequential(
            conv_3x3_bn(c_in=backbone_channels[0], c_out=backbone_channels[1], stride=2),
            BasicBlock(c_in=backbone_channels[1], c_out=backbone_channels[1], c_mid=64),
        )
        self.attention = SpatialAttention(3)

    def forward(self, x):
        f_x4 = self.convs_1_4(x)
        f_x8 = self.convs_4_8(f_x4)
        att_mask_x8 = self.attention(f_x8)
        att_mask_x4 = F.interpolate(att_mask_x8, f_x4.shape[-2:], mode='bilinear', align_corners=True)
        return f_x4 * att_mask_x4, f_x8 * att_mask_x8, att_mask_x4


class SOAHead(nn.Module):

    def __init__(self, c_x4, c_x8, num_class=19, c_mid=128):
        super().__init__()
        self.conv_fuse = conv_3x3_bn(c_x4 + c_x8, c_mid, 1)
        self.cls = nn.Conv2d(c_mid, num_class, 1)

    def forward(self, f_x4, f_x8):
        f_x8_4 = F.interpolate(f_x8, size=f_x4.shape[-2:], mode='bilinear', align_corners=True)
        x = self.conv_fuse(torch.cat([f_x4, f_x8_4], dim=1))
        x = self.cls(x)
        return x


class FeatureFusion(nn.Module):
    """

    Args:
        c1: in_channel count: int, the sum of channels count for two branches
        c2: middle and out_channel count: int
        reduction: reduction radio for reducing ops: int
    """

    def __init__(self, c1, c2, reduction=1) -> None:
        super().__init__()
        self.conv_1x1 = conv_1x1_bn(c1, c2)

        self.attention = ChannelAttention(c2, reduction)

    def forward(self, x1, x2):
        """
        forward function
        Args:
            x1: feature map from context path: torch.Tensor
            x2: feature map from small object activation path: torch.Tensor,
                has the same 2d shape as x1's.

        Returns:
            f_fuse: fused features from both context path and SOA path:
                qtorch.Tensor, has the same 2d shape as x1's.
        """
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.attention(fm)
        f_fuse = fm + fm * fm_se
        return f_fuse


class SOSNet(BaseModel):

    def __init__(self, backbone: str = 'ResNet-18', class_num: int = 19):
        super().__init__(backbone, class_num)
        self.soa = SmallObjectActivation(c_in=3, backbone_channels=self.backbone.channels)
        self.ffm_x4 = FeatureFusion(self.backbone.channels[0] + self.soa.channels[0],
                                    self.backbone.channels[0], reduction=1)
        self.ffm_x8 = FeatureFusion(self.backbone.channels[1] + self.soa.channels[1],
                                    self.backbone.channels[1], reduction=1)
        self.seg_head = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=class_num,
                                 scales=(1, 2, 3, 6))
        self.aux_head = SOAHead(self.backbone.channels[0], self.backbone.channels[1], num_class=2, c_mid=128)
        self.apply(self._init_weights)

    def forward(self, x):
        # backbone path
        f_x4, f_x8, f_x16, f_x32 = self.backbone(x)  # 1/4, 1/8, 1/16, 1/32

        # small object attention path
        f_soa_x4, f_soa_x8, att_mask_x4 = self.soa(x)  # 1/4

        # feature fusion for two path
        f_ffm_x4 = self.ffm_x4(f_soa_x4, f_x4)
        f_ffm_x8 = self.ffm_x8(f_soa_x8, f_x8)

        out_seg = self.seg_head([f_ffm_x4, f_ffm_x8, f_x16, f_x32])
        # out_seg = self.seg_head([f_x4, f_x8, f_x16, f_x32])
        if out_seg.shape[-2:] != x.shape[-2:]:
            out_seg = F.interpolate(out_seg, x.shape[-2:], mode='bilinear', align_corners=True)
        if self.training:
            out_aux = self.aux_head(f_soa_x4, f_soa_x8)
            if out_aux.shape[-2:] != x.shape[-2:]:
                out_aux = F.interpolate(out_aux, x.shape[-2:], mode='bilinear', align_corners=True)
            return out_seg, out_aux, att_mask_x4
        return out_seg  # , None, att_mask


if __name__ == "__main__":
    net = SOSNet(backbone="MobileNetV3-large", class_num=8)
    net.cuda()
    net.train()
    net = nn.DataParallel(net)
    in_ten = torch.randn((2, 3, 1024, 512)).cuda()
    _y = net(in_ten)
    print(_y[0].size(), _y[1].size(), _y[2].size())
    from semseg.utils.utils import model_summary, init_logger

    init_logger()
    model_summary(net, (1, 3, 1024, 512))

    # from semseg.models.modules.auxiliary import SmallObjectMask
    #
    # a = SmallObjectMask([-1, 5])
    # ax = torch.ones((4, 4))
    # ax[0][0] = -1
    # ax[3][3] = 5
    # print(a(ax))
