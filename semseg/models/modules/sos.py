"""
@Project : semantic-segmentation 
@File    : sos.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2022/12/4 15:34
@e-mail  : liuwa@hnu.edu.cn
"""
import torch
import logging
from torch import nn
from torch.nn import functional as F
from semseg.losses import CrossEntropy


class SOEM(nn.Module):

    def __init__(self, ignore_label=255, ratio=0.1, threshold=0.5) -> None:
        """
        Small object example mining for SOSNet
        Args:
            ignore_label: int, ignore label id in dataset.
            ratio: double, the minimal ratio to calculate the minimal number of samples.
            threshold: double, the samples with loss larger than the threshold will be selected.
        """
        super().__init__()
        self.ignore_label = ignore_label
        self.ratio = ratio
        self.threshold = threshold

    def forward(self, loss: torch.Tensor, labels_seg: torch.Tensor, labels_so: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss: the joint loss, 0 where the ground truth label is ignored.
            labels_seg: the segmentation labels
            labels_so: the small objet labels that indicate where the small objects are.

        Returns:
            loss_hard: the mean value of those hardest mse losses.
        """
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = int(labels_seg[labels_seg != self.ignore_label].numel() * self.ratio)
        loss_flat = loss.contiguous().view(-1)
        labels_so_flat = labels_so.contiguous().view(-1)
        loss_s = loss_flat[labels_so_flat == 1]  # loss for small objects
        loss_l = loss_flat[labels_so_flat == 0]  # loss for large objects
        loss_hard_s = loss_s[loss_s > self.threshold]  # highest loss for small objects
        loss_hard_l = loss_l[loss_l > self.threshold]  # highest loss for large objects

        if loss_hard_s.numel() < n_min:
            if loss_s.numel() <= n_min:
                loss_hard_s = loss_s
            else:
                loss_hard_s, _ = loss_s.topk(n_min)

        if loss_hard_l.numel() < n_min:
            if loss_l.numel() <= n_min:
                loss_hard_l = loss_l
            else:
                loss_hard_l, _ = loss_l.topk(n_min)

        loss_hard = (torch.sum(loss_hard_s) + torch.sum(loss_hard_l)) / (loss_hard_s.numel() + loss_hard_l.numel())

        # return torch.mean(loss)
        return loss_hard


class HierarchicalLoss(nn.Module):

    def __init__(self, num_class: int, small_obj_ids):
        super().__init__()
        self.m_s = torch.zeros((1, num_class, 1, 1), dtype=torch.int)
        for _id in range(num_class):
            if _id in small_obj_ids:
                self.m_s[0][_id][0][0] = 1
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, logits_b, logits_t):
        """
        compute hierarchical loss, set predicted class belonging to its ancestor class
        Args:
            logits_b: torch.FloatTensor, predicted seg logits, shape=(n_batch, n_class, h, w)
            logits_t: torch.LongTensor, predicted small or large logits, shape=(n_batch, h, w)

        Returns:
            loss_hierarchical: torch.FloatTensor
        """
        # v_seg = torch.argmax(logits_seg, dim=1, keepdim=True) - torch.softmax(logits_seg, dim=1)
        # one_hot_seg = (torch.softmax(logits_seg, dim=1) + v_seg.detach())[:,:1,:,:]
        logits_b = torch.softmax(logits_b, dim=1)
        logits_b_s = torch.mul(logits_b, self.m_s)
        logits_b_l = torch.mul(logits_b, 1 - self.m_s)
        logits_b_s_max, _ = torch.max(logits_b_s, dim=1, keepdim=True)
        logits_b_l_max, _ = torch.max(logits_b_l, dim=1, keepdim=True)
        logits_b_2_t = torch.cat([logits_b_s_max, logits_b_l_max], dim=1)

        v_so = torch.argmax(logits_b_2_t, dim=1, keepdim=True) - torch.softmax(logits_b_2_t, dim=1)
        one_hot_b_2_t = torch.softmax(logits_b_2_t, dim=1) + v_so.detach()

        v_so = torch.argmax(logits_t, dim=1, keepdim=True) - torch.softmax(logits_t, dim=1)
        one_hot_t = torch.softmax(logits_t, dim=1) + v_so.detach()

        mse = torch.sum(self.mse(one_hot_b_2_t, one_hot_t), dim=1)
        return mse

    def to(self, device):
        super().to(device)
        self.m_s = self.m_s.to(device)
        return self


class HierarchicalSegLoss(nn.Module):

    def __init__(self, loss_seg_fn, loss_hier_fn, ignore_label, is_hier=False, is_soem=False, ratio=0.1):
        super(HierarchicalSegLoss, self).__init__()
        self.loss_bottom_fn = loss_seg_fn
        self.loss_top_fn = loss_seg_fn
        self.loss_hier_fn = loss_hier_fn
        self.is_hier = is_hier
        self.is_soem = is_soem
        if is_soem:
            self.soem = SOEM(ignore_label=ignore_label, ratio=ratio, threshold=2.5 if self.is_hier else 0.5)

    def forward(self, logits_bottom, logits_top, lbl_bottom, lbl_top):
        """
        compute hierarchical segmentation loss
        Args:
            logits: torch.FloatTensor, predicted logits of sosnet, shape=(n_batch, n_classes+2, h, w)
            lbl_bottom:  torch.FloatTensor, labels for seg mask, shape=(n_batch, h, w)
            lbl_top:  torch.FloatTensor, labels for small object mask, shape=(n_batch, h, w)

        Returns:
            loss_hierSeg: torch.Tensor, a float scaler
        """
        loss_bottom = self.loss_bottom_fn(logits_bottom, lbl_bottom)
        if self.is_hier:
            loss_top = self.loss_top_fn(logits_top, lbl_top)
            loss_hier = self.loss_hier_fn(logits_bottom, logits_top)
            # loss_hier = torch.mean(self.loss_hier_fn(logits_bottom, logits_top))
            # return loss_hier
        else:
            loss_top = 0
            loss_hier = 0

        loss_hierSeg = loss_top + loss_hier + loss_bottom
        if self.is_soem:
            loss_hierSeg = self.soem(loss=loss_hierSeg, labels_seg=lbl_bottom, labels_so=lbl_top)
        else:
            loss_hierSeg = torch.mean(loss_hierSeg)

        return loss_hierSeg


if __name__ == "__main__":
    _pred = torch.rand(5, 4, 480, 640, dtype=torch.float)
    mmmm = torch.ones_like(_pred, dtype=torch.float, requires_grad=True)
    _lb = torch.rand(5, 2, 480, 640, dtype=torch.float)
    _loss_f = HierarchicalLoss(num_class=4, small_obj_ids=[1, 2])
    _loss = _loss_f(_pred * mmmm, _lb)
    _loss.requires_grad_(True)
    print(mmmm.grad)
    torch.mean(_loss).backward()
    print(mmmm.grad)
    print(_loss.size())

