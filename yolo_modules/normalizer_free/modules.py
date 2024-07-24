import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, cv1, cv2, add):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        # c_ = int(c2 * e)  # hidden channels
        self.cv1 = cv1
        self.cv2 = cv2
        
        self.add = add
        
        self.alpha = 0.2
        self.gain = Parameter(torch.ones(()))
        
        self.beta = 1
        

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return self.alpha * x + self.cv2(self.cv1(x)) * self.gain / self.beta if self.add else self.cv2(self.cv1(x))