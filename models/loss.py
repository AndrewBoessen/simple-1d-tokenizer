"""
Loss Modules
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1).eval()
        self.register_buffer("imagenet_mean", torch.Tensor(
            _IMAGENET_MEAN)[None, :, None, None])
        self.register_buffer("imagenet_std", torch.Tensor(
            _IMAGENET_STD)[None, :, None, None])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Always in eval mode.
        self.eval()

        input = torch.nn.functional.interpolate(
            input, size=224, mode="bilinear", align_corners=False, antialias=True)
        target = torch.nn.functional.interpolate(
            target, size=224, mode="bilinear", align_corners=False, antialias=True)
        pred_input = self.convnext(
            (input - self.imagenet_mean) / self.imagenet_std)
        pred_target = self.convnext(
            (target - self.imagenet_mean) / self.imagenet_std)
        loss = torch.nn.functional.mse_loss(
            pred_input,
            pred_target,
            reduction="mean")

        return loss
