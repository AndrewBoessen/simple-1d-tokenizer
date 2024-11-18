"""
Image Tokenizer Decoder
"""

import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn

from .blocks import ResidualAttention


class RemoveLatentTokens(nn.Module):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, x):
        return x[:, 0 : self.grid_size**2]


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        self.width = {
            "small": 128,
            "base": 768,
            "large": 1024,
        }[self.model_size]
        self.num_layers = {
            "small": 1,
            "base": 12,
            "large": 24,
        }[self.model_size]
        self.num_heads = {
            "small": 8,
            "base": 12,
            "large": 16,
        }[self.model_size]

        # project token dim to model dim
        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        scale = self.width**-0.5  # scale by 1 / sqrt(d)
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size**2, self.width)
        )
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width)
        )
        self.ln_pre = nn.LayerNorm(self.width)  # pre attention layer norm
        self.transformer = nn.ModuleList()  # attention layers
        for _ in range(self.num_layers):
            self.transformer.append(
                ResidualAttention(self.width, self.num_heads, mlp_ratio=4.0)
            )
        self.ln_post = nn.LayerNorm(self.width)  # post attention layer norm
        # FFN to convert mask tokens to image patches
        self.ffn = nn.Sequential(
            nn.Conv2d(self.width, 3 * self.patch_size**2, 1, padding=0, bias=True),
            Rearrange(
                "B (P1 P2 C) H W -> B C (H P1) (W P2)",
                P1=self.patch_size,
                P2=self.patch_size,
            ),
        )
        # conv layer on pixel output
        self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)

        self.model = nn.Sequential(
            self.ln_pre,
            Rearrange("B L C -> L B C"),
            *self.transformer,
            Rearrange("L B C -> B L C"),
            RemoveLatentTokens(grid_size=self.grid_size),
            self.ln_post,
            Rearrange("B (H W) C -> B C H W", H=self.grid_size, W=self.grid_size),
            self.ffn,
            self.conv_out
        )

    def forward(self, z_q):
        B, _, L = z_q.shape  # Batch, Channels, Length
        assert L == self.num_latent_tokens

        x = einops.rearrange(z_q, "B C L -> B L C")
        x = self.decoder_embed(x)  # embed tokens in model dim

        mask_tokens = self.mask_token.repeat(B, self.grid_size**2, 1).to(x.dtype)
        # Add positional embeddings
        mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
        x = x + self.latent_token_positional_embedding[:L]
        x = torch.cat([mask_tokens, x], dim=1)

        x = self.model(x)  # decode latent tokens

        return x
