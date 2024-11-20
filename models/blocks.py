"""
Transformer Attention Blocks
"""

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import einops


class Residual(nn.Module):
    """
    Residual Convolutional Layer
    """

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        """
        initialize residual CNN layer

        :param in_channels number: Number of input channels
        :param num_hiddens number: Number of hidden channels
        :param num_residual_hiddens number: Number of residual hiddens
        """
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            # C: 3 -> residual hidden
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            # C: resiual hidden -> out_hidden
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        """
        Residual layer

        :param x numpy.ndarray: Input image
        """
        return x + self._block(x)  # residual output


class ResidualStack(nn.Module):
    """
    Residual Convolution Stack
    """

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        """
        initialize residual stack

        :param in_channels number: Number of input channels
        :param num_hiddens number: Number of hidden channels
        :param num_residual_layers number: Number of residual layers in stack
        :param num_residual_hiddens number: Number of hidden residual channels
        """
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        """
        Apply residual stack

        :param x numpy.ndarray: Input image
        """
        # Apply all residual layers in stack
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class ResidualAttention(nn.Module):
    """
    Residual Attention Block
    """

    def __init__(
        self,
        d_model,
        n_head,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        """
        Initialize attention blocks params and weight

        :param d_model int: model dimension
        :param n_head int: number of heads
        :param mlp_ratio float: ratio of FFN hidden to input dim
        :param act_layer nn.Module: activation function to use
        :param norm_layer nn.Module: normalization function to use
        """
        super().__init__()

        self.ln_1 = norm_layer(d_model)  # first layer norm
        self.attn = nn.MultiheadAttention(d_model, n_head)  # self attention
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)  # second layern norm before FFN
            mlp_width = int(d_model * mlp_ratio)  # hidden dim in FFN
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
        self,
        x: torch.Tensor
    ):
        """
        Calculate attention values

        :param self class: self
        :param x: input values
        """
        # multihead attention
        # self attention so q, k, v all come from same input
        # dont return weights to enable flash attention
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
        self,
        x: torch.Tensor,
    ):
        """
        Forward pass of residual attention

        :param self class: self
        :param x: input
        """
        # norm and apply attention
        attn_output = self.attention(x=self.ln_1(x))
        # residual connection
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))  # norm and apply residual FFN
        return x


class Attention(nn.Module):
    """
    Attention without residual connection
    """

    def __init__(self, dim, num_heads=8, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Initialize attention params and weights

        :param dim int: model dimension
        :param num_heads int: number of heads
        :param qk_scale float: constant to scale by
        :param attn_drop float: dropout ratio
        :param proj_drop float: output dropout ratio
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # divide model dim between heads
        self.scale = qk_scale or head_dim ** -0.5  # scale to divide by
        # QKV embedding weights
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Apply attention to input

        :param x torch.Tensor: input
        """
        qkv = self.qkv(x)  # QKV projection
        q, k, v = einops.rearrange(
            # split into individual embeddings
            qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v)  # use flash attention
        x = einops.rearrange(x, 'B H L D -> B L (H D)')  # concat heads
        x = self.proj(x)  # output projection
        x = self.proj_drop(x)  # apply dropout
        return x
