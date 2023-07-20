import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None,
                 aux_weights: [tuple, list] = (1, 0.4, 0.4)) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7,
                 aux_weights: [tuple, list] = (1, 1)) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 5
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


# class SOEM(nn.Module):
#
#     def __init__(self, ignore_label=255, ratio=0.1, threshold=0.5) -> None:
#         """
#         Small object example mining for SOSNet
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
#     def forward(self, loss: Tensor, labels: Tensor, labels_s: Tensor) -> Tensor:
#         """
#         Args:
#             loss: the joint loss, 0 where the ground truth label is ignored.
#             labels: the segmentation labels
#             labels_s: the small objet labels that indicate where the small objects are.
#
#         Returns:
#             loss_hard: the mean value of those hardest mse losses.
#         """
#         # preds in shape [B, C, H, W] and labels in shape [B, H, W]
#         n_min = int(labels[labels != self.ignore_label].numel() * self.ratio)
#         loss_flat = loss.contiguous().view(-1)
#         labels_s_flat = labels_s.contiguous().view(-1)
#         loss_s = loss_flat[labels_s_flat == 1]
#         loss_l = loss_flat[labels_s_flat == 0]
#         loss_hard_s = loss_s[loss_s > self.threshold]
#         loss_hard_l = loss_l[loss_l > self.threshold]
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


class BinaryDiceLoss(torch.nn.Module):

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, logits, targets):
        """
        Dice loss for binary segmentation.
        Note that the logits can't be activated before calculating this loss.
        Args:
            logits: torch.FloatTensor, predicted probabilities without sigmoid, shape=(n_batch, h, w)
            targets: torch.LongTensor, ground truth probabilities, shape=(n_batch, h, w)
        Returns:
            score: torch.FloatTensor, dice loss, shape=(1,)
        """
        num = targets.size(0)  # batch size
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: [tuple, list] = (1, 0.4, 0.4)):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels * preds, dim=(2, 3))
        fn = torch.sum(labels * (1 - preds), dim=(2, 3))
        fp = torch.sum((1 - labels) * preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


class Focal(torch.nn.Module):
    def __init__(self, ignore_index=255, weight=None, gamma=2, alpha=None, reduction='none'):
        super(Focal, self).__init__()
        self.ignore_label = ignore_index
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha]).cuda()
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        self.reduction = reduction

    def forward(self, logits, target):
        bs, h, w = target.shape
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # N,C,H,W => N,C,H*W
            logits = logits.transpose(1, 2)  # N,C,H*W => N,H*W,C
            logits = logits.contiguous().view(-1, logits.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)
        logpt = torch.log_softmax(logits, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss.reshape(-1, h, w)
        target = target.reshape(-1, h, w)
        mean_loss = loss.mean().detach()
        loss = torch.where(target != self.ignore_label, loss, mean_loss)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# class Focal(nn.Module):
#     def __init__(self, ignore_index=255, weight=None, gamma=2, size_average=True):
#         super(Focal, self).__init__()
#         self.gamma = gamma
#         self.size_average = size_average
#         self.CE_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
#
#     def forward(self, output, target):
#         logpt = self.CE_loss(output, target)
#         pt = torch.exp(-logpt)
#         loss = ((1 - pt) ** self.gamma) * logpt
#         return loss
#         # if self.size_average:
#         #     return loss.mean()
#         # return loss.sum()


class FocalDice(nn.Module):
    def __init__(self, ignore_index=255, weight=None, gamma=2, size_average=True,
                 delta: float = 0.5, aux_weights: [tuple, list] = (1, 0.4, 0.4)):
        super(FocalDice, self).__init__()
        self.focal = Focal(ignore_index, weight, gamma, size_average)
        self.dice = Dice(delta, aux_weights)

    def forward(self, output, target):
        return self.focal(output, target) + self.dice(output, target)


class DiceBCELoss(nn.Module):

    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = BinaryDiceLoss()
        self.bce = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits, targets):
        """
        A loss combine binary dice loss and binary cross-entropy loss.
        Note that the logits can't be activated before calculating this loss.
        Args:
            logits: torch.FloatTensor, predicted probabilities without sigmoid, shape=(n_batch, h, w)
            targets: torch.LongTensor, ground truth probabilities, shape=(n_batch, h, w)
        Returns:
            loss_diceBce
        """
        loss_dice = self.dice(logits, targets)
        loss_bce = self.bce(logits, targets)
        loss_diceBce = loss_dice + loss_bce
        return loss_diceBce


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'Focal', 'FocalDice', 'DiceBCELoss', 'get_loss']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\n" \
                                    f"Available loss functions: {__all__} "
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    _pred = torch.randint(0, 2, (2, 3, 480, 640), dtype=torch.float).cuda()
    _label = torch.randint(0, 3, (2, 480, 640), dtype=torch.long).cuda()
    _pred2 = torch.randint(0, 2, (2, 480, 640), dtype=torch.float).cuda()
    _label2 = torch.randint(0, 2, (2, 480, 640), dtype=torch.float).cuda()
    loss_fn = Focal(ignore_index=0)
    y = loss_fn(_pred, _label)
    print(y)
