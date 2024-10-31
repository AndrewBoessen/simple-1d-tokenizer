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


class NLayerDiscriminator(nn.Module):
    """
    GAN Discriminator
    """

    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 128,
        num_stages: int = 3,
        blur_resample: bool = True,
        blur_kernel_size: int = 4
    ):
        """
        Initialize discriminator modules

        :param num_channels: channels in pixel space
        :param hidden_channels: hidden dim of conv net
        :param num_stages: number of blocks in conv net
        :param blur_resample: use blur for pool
        :param blur_kernel_size: in kernel size
        """
        super().__init__()
        assert num_stages > 0, "connot have 0 stages in discriminator"

        # channel multipliers
        in_channel_mult = (1,) + tuple(map(lambda t: 2**t,
                                           range(num_stages)))  # (1, 1, 2, 4)
        activation = functools.partial(nn.LeakyReLU, negative_slope=0.1)

        # convert pixel space to hidden dim
        self.block_in = nn.Sequential(
            Conv2dSame(
                num_channels,
                hidden_channels,
                kernel_size=5
            ),
            activation(),
        )

        # blur kernels used in pooling
        BLUR_KERNEL_MAP = {
            3: (1, 2, 1),
            4: (1, 3, 3, 1),
            5: (1, 4, 6, 4, 1),
        }

        discriminator_blocks = []
        for i in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i]  # d * 2^i
            out_channels = hidden_channels * \
                in_channel_mult[i + 1]  # d * 2^(i+1)
            block = nn.Sequential(
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3
                ),
                nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample else BlurBlock(
                    BLUR_KERNEL_MAP[blur_kernel_size]),
                nn.GroupNorm(32, out_channels),
                activation(),  # leakyRelu
            )
            discriminator_blocks.append(block)

        self.blocks = nn.ModuleList(discriminator_blocks)
        self.pool = nn.AdaptiveAvgPool2d((16, 16))

        self.to_logits = nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply discriminator to image

        :param x: input image
        :return: discriminator predictions
        """
        hidden_states = self.block_in(x)  # pixel -> hidden dim : 5 x 5 kernel
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.pool(hidden_states)  # blur kernels

        return self.to_logits(hidden_states)  # hidden dim -> 1 (T/F bool)
