"""
Loss Modules
"""
from typing import Mapping, Text, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from einpos import rearrange

from .discriminator import NLayerDiscriminator

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(
        torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(
        torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss module
    """

    def __init__(self):
        """
        Initialize loss modules
        """
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
        """
        Compute perceptual loss for module

        :param input: reconstructed image
        :param target: original image
        """
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


class ReconstructionLoss(nn.Module):
    """
    Loss for reconstruction network
    """

    def __init__(self, config):
        """
        Initialize loss modules

        :param config Dict: model config dict
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss().eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int,
        mode: str = "generator"
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        Compute loss for autoencoder

        :param inputs: original image
        :param reconstructions: reconstructed image
        :param extra_result_dict: vq loss results
        :param global_step: index of current step
        :param mode: what to compute loss on, generator or discriminator
        :return: loss values
        :raises ValueError: errors
        """
        inputs = inputs.float()
        reconstructions = reconstructions.float()
        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step: int):
        """
        Find if deiscriminator step

        :param global_step: index of current training step
        """
        return global_step >= self.discriminator_iter_start

    def _forward_generator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        extra_result_dict: Mapping[Text, torch.Tensor],
        global_step: int
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        Compute loss for generator

        :param inputs: input image
        :param reconstructions: reconstructed image
        :param extra_result_dict: vq loss results
        :param global_step: index of train step
        :return: loss value
        :raises ValueError: error
        """
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(
                inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(
                inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {
                             self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(
            global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight *
                             perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(
                d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        global_step: int,
    ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        Compute loss for discriminator training step

        :param inputs: orginal image
        :param reconstructions: resconstructed image
        :param global_step: index of current train step
        :return: loss values
        """
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(
            global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * \
            hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + \
                torch.mean(logits_real).detach() * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + \
                torch.mean(logits_fake).detach() * (1 - self.lecam_ema_decay)

        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict
