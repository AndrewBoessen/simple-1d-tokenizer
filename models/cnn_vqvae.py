import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResidualStack


class VectorQuantizeEMA(nn.Module):
    """
    Exponential Moving Average (EMA) vector quantization for VQ-VAE model
    """

    def __init__(
        self, n_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        """
        initialize VQ class

        :param n_embeddings number: number of discrete embeddings
        :param embedding_dim number: dimension of embeddings
        :param commitment_cost number: commitment cost weight
        :param decay number: decay rate for EMA
        :param epsilon number: epsilon value for EMA
        """
        super(VectorQuantizeEMA, self).__init__()

        self._embedding_dim = embedding_dim  # Dimension of an embedding vector, D
        self._n_embeddings = n_embeddings  # Number of categories in distribution, K

        # Parameters
        self._embedding = nn.Embedding(
            self._n_embeddings, self._embedding_dim
        )  # Embedding table for categorical distribution
        self._embedding.weight.data.normal_()  # Randomly initialize embeddings
        self.register_buffer(
            "_ema_cluster_size", torch.zeros(n_embeddings)
        )  # Clusters for EMA
        self._ema_w = nn.Parameter(
            torch.Tensor(n_embeddings, self._embedding_dim)
        )  # EMA weights
        self._ema_w.data.normal_()

        # Loss / Training Parameters
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, z_e):
        """
        Quantize embeddings

        :param z_e numpy.ndarray: Embeddings from encoder to quantize
        """
        # reshape from BCHW -> BHWC
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        shape = z_e.shape

        # Flatten input embeddings
        flat_z_e = z_e.view(-1, self._embedding_dim)

        # Claculate Distances
        # ||z_e||^2 + ||e||^2 - 2 * z_q
        distances = (
            torch.sum(flat_z_e**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_z_e, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._n_embeddings, device=z_e.device
        )
        # Convert to shape of embeddings
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and Unflatten
        z_q = torch.matmul(encodings, self._embedding.weight).view(shape)

        # Update weights with EMA
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._n_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_z_e)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(
            z_q.detach(), z_e
        )  # distance from encoder output and quantized embeddings
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Loss
        z_q = z_e + (z_q - z_e).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # Convert shape back to BCHW
        return loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Encoder(nn.Module):
    """
    Convolutional Encoder producing a 32x32 grid from 256x256 input
    """

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        """
        Initialize encoder network
        :param in_channels: Number of input channels
        :param num_hiddens: Number of hidden channels
        :param num_residual_layers: Number of layers in residual stack
        :param num_residual_hiddens: Number of channels in residual hidden layer
        """
        super(Encoder, self).__init__()

        # Modified convolution layers to achieve 32x32 output from 256x256 input
        self._conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,  # 256 -> 128
        )
        self._conv2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,  # 128 -> 64
        )
        self._conv3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,  # 64 -> 32
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        """
        Encode image
        :param inputs: images to encode (256x256)
        :return: latent representation (32x32)
        """
        x = self._conv1(inputs)
        x = F.relu(x)
        x = self._conv2(x)
        x = F.relu(x)
        x = self._conv3(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    """
    Convolutional Decoder reconstructing 256x256 from 32x32 input
    """

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        """
        Initialize decoder network
        :param in_channels: Number of input channels
        :param num_hiddens: Number of hidden channels
        :param num_residual_layers: Number of residual layers in stack
        :param num_residual_hiddens: Number of channels in residual
        """
        super(Decoder, self).__init__()

        self._conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        # Modified upsampling convolution transpose layers
        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,  # 32 -> 64
        )
        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens // 4,
            kernel_size=4,
            stride=2,
            padding=1,  # 64 -> 128
        )
        self._conv_trans_3 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 4,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,  # 128 -> 256
        )

    def forward(self, inputs):
        """
        Decode latent embeddings
        :param inputs: latent embeddings (32x32)
        :return: reconstructed image (256x256)
        """
        x = self._conv1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        x = F.relu(x)
        return self._conv_trans_3(x)
