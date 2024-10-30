"""
Discriminator for GAN
"""

import functools
import math
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dSame(nn.Conv2d):
    """
    Conv2d that retains image shape
    """

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """
        Find the padding to ensure same sizes

        :param i: input size
        :param k: kernel size
        :param s: stride
        :param d: dialation
        :return: padding
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        2D Convolution

        :param x: input tensor
        :return: conv output
        """
        input_h, input_w = input.shape[-2:]

        pad_h = self.calc_same_pad(
            i=input_h, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(
            i=input_w, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            input = F.pad(input, [pad_w // 2, pad_w - pad_w //
                                  2, pad_h // 2, pad_h - pad_h // 2])  # pad left, right, top, and bottom
        return super().forward(input)
