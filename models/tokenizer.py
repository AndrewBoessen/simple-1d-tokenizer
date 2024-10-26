"""
1D Image Tokenizer
"""

import torch
from torch import nn
from einops import rearrange

from encoder import Encoder
from decoder import Decoder
from quantizer import VectorQuantizer


class Tokenizer(nn.Module):
    """
    1D Image Tokenizer
    """

    def __init__(self, config):
        """
        Initialize tokenizer params

        :param config dict: model config options
        """
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width))

        self.apply(self._init_weights)

        self.quantizer = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            embedding_dim=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
        )

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        """
        Encode Image

        :param x torch.Tensor: pixel values
        """
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_q, result_dict = self.quantizer(z)

        return z_q, result_dict

    def decode(self, z_q):
        """
        Decode Latent Embeddings

        :param z_q torch.Tensor: latent embeddings
        """
        return self.decoder(z_q)

    def decode_tokens(self, tokens):
        """
        Decode Tokens

        :param tokens torch.Tensor: ids of tokens to decode
        """
        tokens = tokens.squeeze(1)
        B, L = tokens.shape
        z_q = self.quantizer.get_codebook_entry(
            tokens.reshape(-1)
        ).reshape(B, L, -1)
        z_q = rearrange(z_q, 'B L C -> B C L')
        return self.decoder(z_q)

    def forward(self, x):
        """
        Forward pass though encoder, quantizer, and decoder

        :param x torch.Tensor: pixel values
        """
        z_q, result_dict = self.encode(x)
        decoded = self.decode(z_q)
        return decoded, result_dict
