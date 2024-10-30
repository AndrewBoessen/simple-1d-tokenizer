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


class BlurBlock(nn.Module):
    """
    Apply Gaussian-like blur to input tensors.

    The blur is implemented as a separable convolution with a customizable kernel.
    The module automatically handles padding to maintain spatial dimensions while
    performing stride-2 downsampling.
    """

    def __init__(self, kernel: Tuple[int] = (1, 3, 3, 1)) -> None:
        """
        Initialize blur module

        :param kernel: The 1D kernel used to create the 2D conv kernel
        """
        super().__init__()

        # Create 2D kernel from 1D separable kernel
        kernel_tensor = torch.tensor(
            kernel,
            dtype=torch.float32,
            requires_grad=False
        )
        kernel_2d = kernel_tensor[None, :] * kernel_tensor[:, None]

        # Normalize kernel
        kernel_2d = kernel_2d / kernel_2d.sum()

        # Add batch and channel dimensions: (1, 1, kernel_size, kernel_size)
        kernel_4d = kernel_2d.unsqueeze(0).unsqueeze(0)

        # Register the kernel as a buffer (non-trainable parameter)
        self.register_buffer("kernel", kernel_4d)

    @staticmethod
    def calc_same_pad(i: int, k: int, s: int) -> int:
        """
        Calculate padding needed to maintain spatial dimensions with given stride.

        :param i: Input dimension size
        :param k: Kernel size
        :param s: Stride

        :return: Required padding size
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply blur kernel to image

        :param x: input image to blur
        :return: blurred image
        """
        # Get input spatial dimensions
        _, channels, height, width = x.shape

        # Calculate padding
        pad_h = self.calc_same_pad(i=height, k=4, s=2)
        pad_w = self.calc_same_pad(i=width, k=4, s=2)

        # Apply padding if needed
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2
                ]
            )

        # Expand kernel to match input channels
        weight = self.kernel.expand(channels, -1, -1, -1)

        # Apply convolution with stride 2
        return F.conv2d(
            input=x,
            weight=weight,
            stride=2,
            groups=channels
        )
