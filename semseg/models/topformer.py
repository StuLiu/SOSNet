"""
@Project : semantic-segmentation
@File    : topformer.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/6/17 下午8:51
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from semseg.models.backbones.topformer import ConvModule, TokenPyramidTransformer, topformer_cfgs
from semseg.models.base import BaseModel
from semseg.models.heads.upernet import UPerHead


class SimpleHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=19, channel=64, dropout_ratio=0.1, is_dw=False):
        super().__init__()
        self.num_classes = num_classes
        self.selected_id = [1, 2, 3]

        if dropout_ratio is not None and dropout_ratio > 0 :
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.linear_fuse = ConvModule(
            channel,
            channel,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=nn.BatchNorm2d,
            act_cfg=nn.ReLU
        )
        self.conv_seg = nn.Conv2d(channel, self.num_classes, kernel_size=(1, 1))

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)
        x = self.agg_res(xx)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        return x

    def _transform_inputs(self, inputs):
        return [inputs[i] for i in self.selected_id]

    def agg_res(self, preds):
        outs = torch.zeros_like(preds[0], requires_grad=True).cuda()
        for pred in preds:
            pred = F.interpolate(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs = outs + pred
        return outs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output


class TopFormer(BaseModel):

    def __init__(self, backbone: str = 'TokenPyramidTransformer-B', num_classes: int = 19) -> None:
        super().__init__('None', num_classes)
        _backbone, _variant = backbone.split('-')
        assert _backbone == 'TokenPyramidTransformer' and _variant in ['B', 'S', 'T']
        self.backbone = eval(_backbone)(**topformer_cfgs[_variant])

        self.head_bootom = SimpleHead(num_classes=num_classes,
                                      channel=self.backbone.out_channels[-1],
                                      dropout_ratio=0.1)
        # self.head_top = SimpleHead(num_classes=2,
        #                            channel=self.backbone.out_channels[-1],
        #                            dropout_ratio=0.1)
        self.head_top = UPerHead(in_channels=self.backbone.out_channels,
                                 channel=32,
                                 num_classes=2,
                                 scales=(1, 2, 3, 6))

    def init_pretrained(self, pretrained: str = None, strict=False) -> None:
        if pretrained and isinstance(self.backbone, nn.Module):
            self.backbone.load_pretrained(pretrained, strict=strict)

    def forward(self, x: torch.Tensor):
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
    _model = TopFormer('TokenPyramidTransformer-B', num_classes=8)
    _model.init_pretrained('../../checkpoints/backbones/topformer/topformer-B-224-75.3.pth')
    _model.train(True).cuda()
    _x = torch.randn(4, 3, 512, 512).cuda()
    _y = _model(_x)
    if _model.training:
        print(_y[0].shape, _y[1].shape)
    else:
        print(_y.shape)
