"""
@Project : semantic-segmentation 
@File    : auxiliary.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/6/6 下午5:13
@e-mail  : 1183862787@qq.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallObjectMask(nn.Module):

    def __init__(self, small_obj_cls_list: list):
        super().__init__()
        self.sol = small_obj_cls_list

    def forward(self, lbl: torch.Tensor) -> torch.Tensor:
        ones, so_mask = torch.ones_like(lbl, dtype=torch.long) * 1, torch.zeros_like(lbl, dtype=torch.long)
        for cls in self.sol:
            so_mask = torch.where(lbl == cls, ones, so_mask)
        return so_mask


class EdgeMask(nn.Module):

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), 1, 1).requires_grad_(False)

    def forward(self, lbl: torch.Tensor) -> torch.Tensor:
        avg_lbl = self.avg_pool(lbl.float())
        delta = torch.abs(lbl - avg_lbl)
        edge_mask = torch.zeros_like(lbl, dtype=torch.long)
        edge_mask[delta > 0.1] = 1
        return edge_mask


class EdgeMask2(nn.Module):
    def __init__(self):
        super().__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False)  # .type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6. / 10], [3. / 10], [1. / 10]], dtype=torch.float32).reshape(1, 3, 1, 1)
        )

    def to(self, device):
        super().to(device)
        self.laplacian_kernel = self.laplacian_kernel.to(device)
        return self

    def forward(self, gtmasks):
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).float(), self.laplacian_kernel, padding=1)
        boundary_targets = torch.clamp(boundary_targets, min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).float(), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = torch.clamp(boundary_targets_x2, min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).float(), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = torch.clamp(boundary_targets_x4, min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).float(), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = torch.clamp(boundary_targets_x8, min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        return torch.squeeze(boudary_targets_pyramid, dim=1)


if __name__ == '__main__':

    from semseg.datasets.isaid import ISAID
    from semseg.datasets.vaihingen import Vaihingen
    from torch.utils.data import DataLoader
    import cv2
    import numpy as np

    # _dataset = ISAID('../../../data/iSAID2')
    # _loader = DataLoader(_dataset, 16, shuffle=True)
    # _transfer_so = SmallObjectMask(ISAID.SMALL_OBJECT)
    _dataset = Vaihingen('../../../data/ISPRS_DATA/Vaihingen2')
    _loader = DataLoader(_dataset, 8, shuffle=True)
    _transfer_so = SmallObjectMask(Vaihingen.SMALL_OBJECT)
    _transfer_edge = EdgeMask()  # .cuda()
    for _img, _lbl in _loader:
        _mask_so = _transfer_so(_lbl).detach()
        _mask_edge = _transfer_edge(_lbl).detach()
        for _img_i, _so_i, _edge_i in zip(_img, _mask_so, _mask_edge):
            _so_i = _so_i.numpy().astype(np.uint8)
            _edge_i = _edge_i.numpy().astype(np.uint8)
            _img_i = np.transpose(_img_i.numpy().astype(np.uint8), (1, 2, 0))
            print(np.unique(_edge_i))
            cv2.imshow('0', cv2.cvtColor(_img_i, cv2.COLOR_RGB2BGR))
            cv2.imshow('1', _so_i)
            cv2.imshow('2', _edge_i * 255)
            cv2.waitKey(0)
        break
