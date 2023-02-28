import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead, UPerHead


class SegFormer(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.head_bootom = SegFormerHead(self.backbone.channels,
                                         256 if 'B0' in backbone or 'B1' in backbone else 768,
                                         num_classes)
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                channel=32,
                                num_classes=2,
                                scales=(1, 2, 3, 6))
    def forward(self, x: Tensor):
        f_x4, f_x8, f_x16, f_x32  = self.backbone(x)
        logits_bottom = self.head_bootom([f_x4, f_x8, f_x16, f_x32 ])   # 4x reduction in image size
        logits_bottom = F.interpolate(logits_bottom, size=x.shape[2:], mode='bilinear', align_corners=True)

        if self.training:
            # logits_edge = self.head_edge(f_x4, f_x8)
            # logits_edge = F.interpolate(logits_edge, x.shape[-2:], mode='bilinear', align_corners=True)
            logits_top = self.head_top([f_x4, f_x8, f_x16, f_x32])
            logits_top = F.interpolate(logits_top, x.shape[-2:], mode='bilinear', align_corners=True)
            # return torch.cat([logits_seg, logits_so], dim=1), logits_edge
            return logits_bottom, logits_top, None

        return logits_bottom.contiguous()


if __name__ == '__main__':
    model = SegFormer('MiT-B0', num_classes=8)
    model.train(True)
    model.init_pretrained('../../checkpoints/backbones/mit/mit_b0.pth')
    x = torch.zeros(4, 3, 512, 1024)
    y = model(x)
    if model.training:
        print(y[0].shape, y[1].shape)
    else:
        print(y.shape)