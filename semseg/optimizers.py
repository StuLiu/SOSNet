from torch import nn
from torch.optim import AdamW, SGD

# def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
#
#     # 获得新添加⽹络层的参数
#     backbone_param = list(map(id, model.backbone.parameters()))
#     # 获得预训练模型的参数
#     new_param = filter(lambda p: id(p) not in backbone_param, model.parameters())
#     # 定义优化器和损失函数
#     params = [
#         {'params': model.backbone.parameters(), 'lr': lr * 0.1},
#         {'params': new_param, 'lr': lr * 0.1}
#     ]
#
#     if optimizer == 'adamw':
#         return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
#     else:
#         return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if p.dim() == 1:
            nwd_params.append(p)
        else:
            wd_params.append(p)

    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0},
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)


# params = [
#     {"params": [value] for _, value in model.sharedNet.named_parameters() if value.requires_grad},
#     {"params": [value for _, value in model.cls_fc_son1.named_parameters()
#                 if value.requires_grad], 'lr': args.lr * 10},
#     {"params": [value for _, value in model.cls_fc_son2.named_parameters()
#                 if value.requires_grad], 'lr': args.lr * 10},
#     {"params": [value for _, value in model.sonnet1.named_parameters()
#                 if value.requires_grad], 'lr': args.lr * 10},
#     {"params": [value for _, value in model.sonnet2.named_parameters()
#                 if value.requires_grad], 'lr': args.lr * 10},
# ]
# optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)


# '''
# ⾸先，我们期望预训练模型的学习率是新添加⽹络层学习率的⼗分之⼀。
# 具体的，筛选出新添加⽹络层的参数和预训练模型的参数分别为其配置学习率
#     '''
# # 获得新添加⽹络层的参数
# backbone_param = list(map(id, model.backbone.parameters()))
# # 获得预训练模型的参数
# new_param = filter(lambda p: id(p) not in backbone_param , model.parameters())
# # 定义优化器和损失函数
# optimizer = torch.optim.Adam([
#     {'params': backbone_param, 'lr': LR * 0.1},
#     {'params': new_param}
# ], lr=LR)