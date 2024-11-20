"""
1D Image Tokenizer
"""

import torch
from einops import rearrange
from torch import nn

from .base_model import BaseModel
from .cnn_vqvae import Decoder as PixelDecoder
from .cnn_vqvae import Encoder as PixelEncoder
from .cnn_vqvae import VectorQuantizeEMA
from .decoder import Decoder
from .encoder import Encoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):
    """
    VQ-VAE Network
    Contains: Encoder, Quantizer, Decoder
    """

    def __init__(self, pretrained_weight):
        """
        Initialize VQ-VAE encoder decoder network

        :param num_hiddens number: Number of hidden layers
        :param num_residual_layers number: Number of residual stacks
        :param num_residual_hiddens number: Number of channels in hidden layer
        :param num_embeddings number: Number of discrete embeddings
        :param embedding_dim number: Dimension of discrete embeddings
        :param commitment_cost number: Weight for commitment const in loss
        :param decay number: Decay parameter in EMA
        """
        num_hiddens = 128
        num_residual_layers = 2
        num_residual_hiddens = 32
        num_embeddings = 128
        embedding_dim = 8
        commitment_cost = 0.25
        decay = 0.99

        super(VQVAE, self).__init__()

        self.load_state_dict(
            torch.load(pretrained_weight, map_location=torch.device("cpu")), strict=True
        )

        self._encoder = PixelEncoder(
            3, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )

        self._vq = VectorQuantizeEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )
        self._decoder = PixelDecoder(
            embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens
        )

        self.eval()
        for param in self.parameters():
            param.requires_grad(False)

    def set_embeddings(self, new_embeddings):
        """
        Set discrete embeddings codebook params

        :param new_embeddings numpy.ndarray: Embedding codebook
        """
        with torch.no_grad():
            self._vq._embedding.weight.copy_(new_embeddings)

    def encode(self, x):
        """
        Encode image

        :param x numpy.ndarray: Input image
        """
        z = self._encoder(x)
        z_e = self._pre_vq_conv(z)
        return z_e

    def pretrain(self, x):
        """
        Bypass vector quantize step for pretraining

        :param x numpy.ndarray: Input image
        """
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        x_recon = self._decoder(z)
        return x_recon

    def forward(self, x):
        """
        Encode and reconstruct image

        :param x numpy.ndarray: Input image
        """
        z = self._encoder(x)  # encode image to latent
        z = self._pre_vq_conv(z)
        # quantize encoding to dicrete space
        loss, z_q, perplexity, _ = self._vq(z)
        x_recon = self._decoder(z_q)  # reconstruction of input from decoder
        return loss, x_recon, perplexity


class Tokenizer(BaseModel):
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

        self.finetune_decoder = config.model.vq_model.get("finetune_decoder", True)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width**-0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.encoder.width)
        )

        self.apply(self._init_weights)

        self.quantizer = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            embedding_dim=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
        )

        if self.finetune_decoder:
            # Freeze encoder and quantizer gradients
            self.latent_tokens.requires_grad(False)
            self.encoder.eval()
            self.encoder.requires_grad(False)
            self.quantizer.eval()
            self.quantizer.requires_grad(False)

            self.pixel_quantizer = VectorQuantizeEMA(
                n_embeddings=64,
                embedding_dim=8,
                commitment_cost=0.25,
                decay=0.99,
                epsilon=0.01,
            )
            self.pixel_decoder = PixelDecoder(
                in_channels=3,
                num_hiddens=128,
                num_residual_layers=2,
                num_residual_hiddens=32,
            )

    def _init_weights(self, module):
        """Initialize the weights.
        :param:
            module -> torch.nn.Module: module to initialize
        """
        if (
            isinstance(module, nn.Linear)
            or isinstance(module, nn.Conv1d)
            or isinstance(module, nn.Conv2d)
        ):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data, mean=0.0, std=0.02
            )
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        """
        Encode Image

        :param x torch.Tensor: pixel values
        """
        if self.finetune_decoder:
            with torch.no_grad():
                self.encoder.eval()
                self.quantizer.eval()
                z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
                z_q, result_dict = self.quantizer(z)
                result_dict["quantizer_loss"] *= 0
                result_dict["commitment_loss"] *= 0
                result_dict["codebook_loss"] *= 0
        else:
            z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
            z_q, result_dict = self.quantizer(z)

        return z_q, result_dict

    def decode(self, z_q):
        """
        Decode Latent Embeddings

        :param z_q torch.Tensor: latent embeddings
        """
        decoded = self.decoder(z_q)
        if self.finetune_decoder:
            quantized_states = torch.einsum(
                "N C H W, C D -> N D H W",
                decoded.softmax(1),
                self.pixel_quantizer._embedding.weight,
            )
            decoded = self.pixel_decoder(quantized_states)
        return decoded

    def decode_tokens(self, tokens):
        """
        Decode Tokens

        :param tokens torch.Tensor: ids of tokens to decode
        """
        tokens = tokens.squeeze(1)
        B, L = tokens.shape
        z_q = self.quantizer.get_codebook_entry(tokens.reshape(-1)).reshape(B, L, -1)
        z_q = rearrange(z_q, "B L C -> B C L")
        return self.decoder(z_q)

    def forward(self, x):
        """
        Forward pass though encoder, quantizer, and decoder

        :param x torch.Tensor: pixel values
        """
        z_q, result_dict = self.encode(x)
        decoded = self.decode(z_q)
        return decoded, result_dict
