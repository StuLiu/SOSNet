"""
@Project : semantic-segmentation 
@File    : upernet.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/5/26 下午3:08
@e-mail  : 1183862787@qq.com
"""

import torch
from torch import nn
from torch.nn import functional as F
from semseg.models.layers import ConvModule
from semseg.models.base import BaseModel
from semseg.models.heads.fcn import FCNHead
from semseg.models.heads.upernet import UPerHead
from semseg.models.modules.attention import SpatialAttention, ChannelAttention, SEModule
from semseg.losses import DiceBCELoss, CrossEntropy



class DetailHead(nn.Module):

    def __init__(self, in_channels_x4, in_channels_x8, mid_channels, n_classes=1):
        super().__init__()
        self.convs_x8 = nn.Sequential(
            nn.Conv2d(in_channels_x8, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
        )
        self.convs_x4 = nn.Sequential(
            nn.Conv2d(in_channels_x4 + mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, n_classes, 3, 1, 1),
            # nn.Sigmoid(),
        )

    def forward(self, f_x4, f_x8):
        up_x4 = F.interpolate(self.convs_x8(f_x8), size=f_x4.shape[-2:], mode='bilinear', align_corners=True)
        f_x4_cat = torch.cat([f_x4, up_x4], dim=1)
        return self.convs_x4(f_x4_cat)


class UperNet(BaseModel):

    def __init__(self, backbone: str = "MobileNetV3-large", class_num: int = 19):
        assert backbone in ["MobileNetV3-large", "MobileNetV3-small"]
        super().__init__(backbone, class_num)
        self.head_edge = DetailHead(in_channels_x4=self.backbone.channels[0],
                                    in_channels_x8=self.backbone.channels[1],
                                    mid_channels=64, n_classes=1)
        self.head_seg = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=class_num,
                                 scales=(1, 2, 3, 6))
        self.head_so = UPerHead(in_channels=self.backbone.channels,
                                channel=32,
                                num_classes=2,
                                scales=(1, 2, 3, 6))
        self.apply(self._init_weights)

    def forward(self, x):
        # backbone path
        f_x4, f_x8, f_x16, f_x32 = self.backbone(x)  # 1/4, 1/8, 1/16, 1/32

        logits_seg = self.head_seg([f_x4, f_x8, f_x16, f_x32])

        if logits_seg.shape[-2:] != x.shape[-2:]:
            logits_seg = F.interpolate(logits_seg, x.shape[-2:], mode='bilinear', align_corners=True)

        if self.training:
            logits_edge = self.head_edge(f_x4, f_x8)
            logits_edge = F.interpolate(logits_edge, x.shape[-2:], mode='bilinear', align_corners=True)
            logits_so = self.head_so([f_x4, f_x8, f_x16, f_x32])
            logits_so = F.interpolate(logits_so, x.shape[-2:], mode='bilinear', align_corners=True)
            # return torch.cat([logits_seg, logits_so], dim=1), logits_edge
            return logits_seg, logits_so, logits_edge
        return logits_seg.contiguous()


if __name__ == "__main__":
    net = UperNet(backbone="MobileNetV3-large", class_num=8)
    net.cuda()
    net.train()
    in_ten = torch.randn((2, 3, 1024, 512)).cuda()
    print(net(in_ten)[0].size(), net(in_ten)[1].size())  # , net(in_ten)[2].size())
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


# class HierarchicalLoss2(nn.Module):
#
#     def __init__(self, small_obj_ids: list):
#         super().__init__()
#         self.small_obj_ids = small_obj_ids
#         self.ce_loss = CrossEntropy()
#         # from torch.nn.modules.loss import BCELoss
#
#     def forward(self, logits_seg, logits_so):
#         """
#         compute hierarchical loss, set predicted class belonging to its ancestor class
#         Args:
#             logits_seg: torch.FloatTensor, predicted seg logits, shape=(n_batch, n_class, h, w)
#             logits_so: torch.LongTensor, predicted small or large logits, shape=(n_batch, h, w)
#
#         Returns:
#             loss_hierarchical: torch.FloatTensor
#         """
#         logits_seg = F.softmax(logits_seg, dim=1)
#         logits_ids = torch.argmax(logits_seg, dim=1)
#         # size_seg_so = (logits_seg.size()[0], 2, logits_seg.size()[1], logits_seg.size()[2])
#         logits_seg_so = torch.zeros_like(logits_ids, dtype=torch.long)
#
#         # set 1 if seg logits indicate the small objects, 0 else
#         for id_i in self.small_obj_ids:
#             logits_seg_so = torch.where(logits_ids == id_i, 1, logits_seg_so)
#         logits_seg_so = logits_seg_so.unsqueeze(dim=1)
#         logits_seg_so = torch.cat([1 - logits_seg_so, logits_seg_so], dim=1)
#         logits_so = torch.argmax(logits_so, dim=1)
#         loss_hierarchical = self.ce_loss(logits_seg_so.float(), logits_so)
#         return loss_hierarchical