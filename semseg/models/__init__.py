from .segformer import SegFormer
# from .ddrnet import DDRNet
from .ddrnet_official import DDRNet
from .fchardnet import FCHarDNet
from .sfnet import SFNet
from .bisenetv1 import BiSeNetv1
from .bisenetv2 import BiSeNetv2
from .lawin import Lawin

# added models
from .deeplabv3plus import DeeplabV3Plus
from .pspnet import PSPNet
from .upernet import UperNet
# from .sosnet_ablation import SOSNetBaseline, SOSNetSB, SOSNetDFEMABL
from .fast_scnn import FastSCNN


__all__ = [
    'SegFormer', 
    'Lawin',
    'SFNet', 
    'BiSeNetv1', 
    
    # Standalone Models
    'DDRNet', 
    'FCHarDNet', 
    'BiSeNetv2',

    # added models
    'FastSCNN',
    'PSPNet',
    'DeeplabV3Plus',
    'UperNet',
    # 'SOSNet', 'SOSNetBaseline', 'SOSNetSB', 'SOSNetDFEMABL',
]
