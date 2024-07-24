import torch
from torch import nn
import warnings

def replace_conv(module: nn.Module, conv_class):
    """Recursively replaces every convolution with WSConv2d.

    Usage: replace_conv(model) #(In-line replacement)
    Args:
      module (nn.Module): target's model whose convolutions must be replaced.
      conv_class (Class): Class of Conv(WSConv2d or ScaledStdConv2d)
    """
    warnings.warn("Make sure to use it with non-residual models only")
    for name, mod in module.named_children():
        target_mod = getattr(module, name)
        if type(target_mod) == torch.nn.Conv2d:
            setattr(module, name, conv_class(target_mod.in_channels, target_mod.out_channels, target_mod.kernel_size,
                                           target_mod.stride, target_mod.padding, target_mod.dilation, target_mod.groups, target_mod.bias is not None))
        
        if type(target_mod) == torch.nn.BatchNorm2d:
            setattr(module, name, torch.nn.Identity())

    for name, mod in module.named_children():
        replace_conv(mod, conv_class)

#  in_channels: int,
#  out_channels: int,
#  kernel_size,
#  stride=1,
#  padding=0,
#  output_padding=0,
#  groups: int = 1,
#  bias: bool = True,
#  dilation: int = 1,
#  padding_mode: str = 'zeros'

def replace_transpose_conv2d(module: nn.Module, conv_class):
    """Recursively replaces every convolution with WSConvTranspose2d.

    Usage: replace_conv(model) #(In-line replacement)
    Args:
      module (nn.Module): target's model whose convolutions must be replaced.
      conv_class (Class): Class of Conv(WSConv2d or ScaledStdConv2d)
    """
    warnings.warn("Make sure to use it with non-residual models only")
    for name, mod in module.named_children():
        target_mod = getattr(module, name)
        if type(target_mod) == torch.nn.ConvTranspose2d:
            setattr(module, name, conv_class(target_mod.in_channels, target_mod.out_channels, target_mod.kernel_size,
                                           target_mod.stride, target_mod.padding, target_mod.output_padding, target_mod.groups, target_mod.bias is not None, target_mod.dilation))
        
        if type(target_mod) == torch.nn.BatchNorm2d:
            setattr(module, name, torch.nn.Identity())

    for name, mod in module.named_children():
        replace_transpose_conv2d(mod, conv_class)


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5

from .modules import Bottleneck
def replace_residual_connection_modules_for_yolo(module: nn.Module, conv_class=Bottleneck):
    try:
        import ultralytics.nn.modules
    except:
        pass
    for name, mod in module.named_children():
        target_mod = getattr(module, name)
        if type(target_mod) == ultralytics.nn.modules.Bottleneck:
            setattr(module, name, conv_class(target_mod.cv1, target_mod.cv2, target_mod.add))

    for name, mod in module.named_children():
        replace_residual_connection_modules_for_yolo(mod, conv_class)