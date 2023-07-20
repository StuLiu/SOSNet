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


#
# class SOEM(nn.Module):
#
#     def __init__(self, ignore_label=255, ratio=0.1, threshold=0.5) -> None:
#         """
#         Small object example mining for UperNet
#         Args:
#             ignore_label: int, ignore label id in dataset
#             ratio:
#             threshold:
#         """
#         super().__init__()
#         self.ignore_label = ignore_label
#         self.ratio = ratio
#         self.threshold = threshold
#
#     def forward(self, loss: torch.Tensor, labels_seg: torch.Tensor, labels_so: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             loss: the joint loss, 0 where the ground truth label is ignored.
#             labels_seg: the segmentation labels
#             labels_so: the small objet labels that indicate where the small objects are.
#
#         Returns:
#             loss_hard: the mean value of those hardest mse losses.
#         """
#         # preds in shape [B, C, H, W] and labels in shape [B, H, W]
#         n_min = int(labels_seg[labels_seg != self.ignore_label].numel() * self.ratio)
#         loss_flat = loss.contiguous().view(-1)
#         labels_so_flat = labels_so.contiguous().view(-1)
#         loss_s = loss_flat[labels_so_flat == 1]  # loss for small objects
#         loss_l = loss_flat[labels_so_flat == 0]  # loss for large objects
#         loss_hard_s = loss_s[loss_s > self.threshold]  # highest loss for small objects
#         loss_hard_l = loss_l[loss_l > self.threshold]  # highest loss for large objects
#
#         if loss_hard_s.numel() < n_min:
#             if loss_s.numel() <= n_min:
#                 loss_hard_s = loss_s
#             else:
#                 loss_hard_s, _ = loss_s.topk(n_min)
#
#         if loss_hard_l.numel() < n_min:
#             if loss_l.numel() <= n_min:
#                 loss_hard_l = loss_l
#             else:
#                 loss_hard_l, _ = loss_l.topk(n_min)
#
#         loss_hard = (torch.sum(loss_hard_s) + torch.sum(loss_hard_l)) / (loss_hard_s.numel() + loss_hard_l.numel())
#
#         # return torch.mean(loss)
#         return loss_hard
#
#
# class HierarchicalLoss(nn.Module):
#
#     def __init__(self, num_class: int, small_obj_ids):
#         super().__init__()
#         self.m_s = torch.zeros((1, num_class, 1, 1), dtype=torch.int)
#         for _id in range(num_class):
#             if _id in small_obj_ids:
#                 self.m_s[0][_id][0][0] = 1
#         self.mse = nn.MSELoss(reduction='none')
#
#     def forward(self, logits_b, logits_t):
#         """
#         compute hierarchical loss, set predicted class belonging to its ancestor class
#         Args:
#             logits_b: torch.FloatTensor, predicted seg logits, shape=(n_batch, n_class, h, w)
#             logits_t: torch.LongTensor, predicted small or large logits, shape=(n_batch, h, w)
#
#         Returns:
#             loss_hierarchical: torch.FloatTensor
#         """
#         # v_seg = torch.argmax(logits_seg, dim=1, keepdim=True) - torch.softmax(logits_seg, dim=1)
#         # one_hot_seg = (torch.softmax(logits_seg, dim=1) + v_seg.detach())[:,:1,:,:]
#         logits_b = torch.softmax(logits_b, dim=1)
#         logits_b_s = torch.mul(logits_b, self.m_s)
#         logits_b_l = torch.mul(logits_b, 1 - self.m_s)
#         logits_b_s_max, _ = torch.max(logits_b_s, dim=1, keepdim=True)
#         logits_b_l_max, _ = torch.max(logits_b_l, dim=1, keepdim=True)
#         logits_b_2_t = torch.cat([logits_b_s_max, logits_b_l_max], dim=1)
#
#         v_so = torch.argmax(logits_b_2_t, dim=1, keepdim=True) - torch.softmax(logits_b_2_t, dim=1)
#         one_hot_b_2_t = torch.softmax(logits_b_2_t, dim=1) + v_so.detach()
#
#         v_so = torch.argmax(logits_t, dim=1, keepdim=True) - torch.softmax(logits_t, dim=1)
#         one_hot_t = torch.softmax(logits_t, dim=1) + v_so.detach()
#
#         mse = torch.sum(self.mse(one_hot_b_2_t, one_hot_t), dim=1)
#         return mse
#
#     def freeze(self):
#         for (_, param) in self.named_parameters():
#             param.requires_grad = False
#         return self
#
#     def unfreeze(self):
#         for (_, param) in self.named_parameters():
#             param.requires_grad = True
#         return self
#
#     def to(self, device):
#         super().to(device)
#         self.m_s = self.m_s.to(device)
#         return self
#
#
# class HierarchicalSegLoss(nn.Module):
#
#     def __init__(self, loss_seg_fn, loss_hier_fn, ignore_label, is_soem=False):
#         super(HierarchicalSegLoss, self).__init__()
#         self.loss_bottom_fn = loss_seg_fn
#         self.loss_top_fn = loss_seg_fn
#         self.loss_hier_fn = loss_hier_fn
#         self.is_soem = is_soem
#         if is_soem:
#             self.soem = SOEM(ignore_label=ignore_label, ratio=0.1, threshold=0.5 if loss_hier_fn is None else 2.5)
#
#     def forward(self, logits, lbl_seg, lbl_so):
#         """
#         compute hierarchical segmentation loss
#         Args:
#             logits: torch.FloatTensor, predicted logits of upernet, shape=(n_batch, n_classes+2, h, w)
#             lbl_seg:  torch.FloatTensor, labels for seg mask, shape=(n_batch, h, w)
#             lbl_so:  torch.FloatTensor, labels for small object mask, shape=(n_batch, h, w)
#
#         Returns:
#             loss_hierSeg: torch.Tensor, a float scaler
#         """
#         loss_bottom = self.loss_bottom_fn(logits[:, :-2, :, :], lbl_seg)
#         if self.loss_hier_fn is not None:
#             loss_top = self.loss_top_fn(logits[:, -2:, :, :], lbl_so)
#             loss_hier = self.loss_hier_fn(logits[:, :-2, :, :], logits[:, -2:, :, :])
#             # loss_hier = torch.mean(self.loss_hier_fn(logits[:, :-2, :, :], logits[:, -2:, :, :]))
#             # return loss_hier
#         else:
#             loss_top = 0
#             loss_hier = 0
#
#         loss_hierSeg = loss_top + loss_hier + loss_bottom
#         if self.is_soem:
#             loss_hierSeg = self.soem(loss=loss_hierSeg, labels_seg=lbl_seg, labels_so=lbl_so)
#         else:
#             loss_hierSeg = torch.mean(loss_hierSeg)
#
#         return loss_hierSeg
#
#
# if __name__ == "__main__":
#     # _loss = HierarchicalSegLoss(loss_base=CrossEntropy(ignore_label=0),
#     #                             small_obj_ids=[1, 2, 3],
#     #                             ignore_label=0,
#     #                             is_hierarch=True,
#     #                             is_soem=True)
#     # import numpy as np
#     #
#     # np.random.seed(21231)
#     # _pr = np.random.randn(2, 19 + 2, 480, 640)
#     # _pred = F.softmax(torch.tensor(_pr, dtype=torch.float), dim=1)
#     # _lb1 = np.ones((2, 480, 640))
#     # _lb2 = np.ones((2, 480, 640))
#     # _label1 = torch.tensor(_lb1, dtype=torch.long)
#     # _label2 = torch.tensor(_lb2, dtype=torch.long)
#     # _y = _loss(_pred, _label1, _label2)
#     # print(_y)
#     #
#     # _pred = np.random.randint(0, 3, (5, 2, 480, 640))
#     # _lb = np.random.randint(0, 3, (5, 2, 480, 640))
#     # _pred = torch.tensor(_pred, dtype=torch.float)
#     # _lb = torch.tensor(_lb, dtype=torch.long)
#     # _loss = HierarchicalLoss([0, 2])
#     # _y = _loss(_pred, _lb)
#     # print(torch.mean(_y))
#     import numpy as np
#     _pred = torch.rand(5, 4, 480, 640, dtype=torch.float)
#     mmmm = torch.ones_like(_pred, dtype=torch.float, requires_grad=True)
#
#     _lb = torch.rand(5, 2, 480, 640, dtype=torch.float)
#     _loss_f = HierarchicalLoss(num_class=4, small_obj_ids=[1, 2])
#     _loss_f2 = HierarchicalLoss2(small_obj_ids=[1, 2])
#     _loss = _loss_f(_pred * mmmm, _lb)
#     _loss.requires_grad_(True)
#     print(mmmm.grad)
#     torch.mean(_loss).backward()
#     print(mmmm.grad)
#     print(_loss.size())
#

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

#
# class UperNet2(BaseModel):
#
#     def __init__(self, backbone: str = "MobileNetV3-large", class_num: int = 19):
#         assert backbone in ["MobileNetV3-large", "MobileNetV3-small"]
#         super().__init__(backbone, class_num)
#         self.n_classes = class_num
#         self.head_edge = DetailHead(in_channels_x4=self.backbone.channels[0],
#                                     in_channels_x8=self.backbone.channels[1],
#                                     mid_channels=64, n_classes=1)
#         self.head_seg = UPerHead(in_channels=self.backbone.channels,
#                                  channel=32,
#                                  num_classes=class_num + 2,
#                                  scales=(1, 2, 3, 6))
#         self.apply(self._init_weights)
#
#     def forward(self, x):
#         # backbone path
#         f_x4, f_x8, f_x16, f_x32 = self.backbone(x)  # 1/4, 1/8, 1/16, 1/32
#
#         logits_all = self.head_seg([f_x4, f_x8, f_x16, f_x32])
#
#         if logits_all.shape[-2:] != x.shape[-2:]:
#             logits_all = F.interpolate(logits_all, x.shape[-2:], mode='bilinear', align_corners=True)
# 
#         if self.training:
#             logits_edge = self.head_edge(f_x4, f_x8)
#             logits_edge = F.interpolate(logits_edge, x.shape[-2:], mode='bilinear', align_corners=True)
#             return logits_all, logits_edge
#         logits_seg = logits_all[:, : self.n_classes, :, :]
#         return logits_seg.contiguous()


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