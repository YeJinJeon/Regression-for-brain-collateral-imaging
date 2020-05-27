import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.group_norm import GroupNorm3d

class FirstConv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels)
            GroupNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.first_conv(x)

if __name__ == '__main__':
    print("Hello")