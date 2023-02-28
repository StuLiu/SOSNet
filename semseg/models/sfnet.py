import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import SFHead, UPerHead


class SFNet(BaseModel):
    def __init__(self, backbone: str = 'ResNetD-18', num_classes: int = 19):
        # assert 'ResNet' in backbone
        super().__init__(backbone, num_classes)
        self.head = SFHead(self.backbone.channels, 128 if '18' or 'MobileNet' in backbone else 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        outs = self.backbone(x)
        out = self.head(outs)
        out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return out


class SFNet0(BaseModel):
    def __init__(self, backbone: str = 'ResNetD-18', num_classes: int = 19):
        # assert 'ResNet' in backbone
        super().__init__(backbone, num_classes)
        self.head_bottom = SFHead(self.backbone.channels, 128 if '18' or 'MobileNet' in backbone else 256, num_classes)
        self.head_top = UPerHead(in_channels=self.backbone.channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))
        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        f_x4, f_x8, f_x16, f_x32 = self.backbone(x)
        logits_bottom = self.head_bottom([f_x4, f_x8, f_x16, f_x32])  # 4x reduction in image size
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
    model = SFNet('MobileNetV3-large')
    model.train()
    model.init_pretrained('../../checkpoints/backbones/mobilenet_/mobilenetv3_large.pth')
    x = torch.randn(2, 3, 512, 1024)
    y = model(x)
    if model.training:
        print(y[0].shape, y[1].shape)
    else:
        print(y.shape)