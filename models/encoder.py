import torch
import torch.nn as nn
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange
from torch.nn.modules import transformer

from blocks import ResidualAttention


class Encoder(nn.Module):
    """
    Image Tokenizer Encoder
    """

    def __init__(self, config):
        """
        Initialize Encoder

        :param config Dict: encoder config dictionary
        """
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        self.width = {
            "small": 512,
            "base": 768,
            "large": 1024,
        }[self.model_size]
        self.num_layers = {
            "small": 8,
            "base": 12,
            "large": 24,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
        }[self.model_size]

        # Split images into patches and embed into latent dim
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.width,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True
        )

        scale = self.width ** -0.5  # scale by 1/sqrt(d)
        # positional embedding for image patches
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size ** 2 + 1, self.width)
        )
        # positional embedding for latent tokens
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )
        # pre layer norm
        self.ln_pre = nn.LayerNorm(self.width)
        # Transformer blocks
        self.transfomer = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer.append(ResidualAttention(
                self.width,
                self.num_heads,
                mlp_ratio=4.0
            ))
        # post trasnformer layer norm
        self.ln_post = nn.LayerNorm(self.width)
        # project model dim to token dim
        self.conv_out = nn.Conv2d(
            self.width, self.token_size, kernel_size=1, bias=True)

        # encoder model
        self.model = nn.Sequential(
            self.ln_pre,
            Rearrange('B L C -> L B C'),
            *self.transformer,
            Rearrange('L B C -> B L C'),
            # remove images patches
            Rearrange('B P T C -> B T C', P=self.grid_size **
                      2 + 1, T=self.num_latent_tokens),
            self.ln_post,
            self.conv_out,
        )

    def forward(self, pixel_values, latent_tokens):
        """
        Encode image into sequence of latent tokens

        :param pixel_values torch.Tensor: pixel values of image to encode
        :param latent_tokens torch.Tensor: Initial sequence of latent tokens
        """
        x = pixel_values
        x = self.patch_embed(x)
        # flatten paches into sequence
        x = einops.rearrange(x, 'B C H W -> B C (H W)')
        # move channel axis to end
        x = einops.rearrange(x, 'B C L -> B L C')
        # add positional embeddings
        x = x + self.positional_embedding.to(x.dtype)

        # add positional embeddings to latent_tokens
        latent_tokens = latent_tokens + \
            self.latent_token_positional_embedding.to(x.dtype)
        # concat patches and latent tokens
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.model(x)

        return x
