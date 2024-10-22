"""
Vector Quantizer
"""

from typing import Mapping, Text, Tuple

import torch
from einops import rearrange


class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 codebook_size: int = 1024,
                 embedding_dim: int = 256,
                 commitment_cost: float = 0.25
                 ):
        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding_table = torch.nn.Embedding(codebook_size, embedding_dim)
        self.embedding_table.weight.data.uniform_(
            -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()  # cast to float if not already
        # move channels axis to end
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        embedding = self.embedding_table.weight  # embedding tensor

        # KNN embedding search
        # d = ||z - e||_2
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) +\
            torch.sum(embedding ** 2, dim=1) -\
            2 * torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        closest_embedding_ids = torch.argmin(d, dim=1)
        z_q = self.embedding_table(closest_embedding_ids).view(z.shape)

        # Calculate loss
        commitment_loss = self.commitment_cost * \
            torch.mean((z_q.detach() - z)**2)
        codebook_loss = torch.mean((z_q - z.detach())**2)
        loss = commitment_loss + codebook_loss

        # pass through gradients from z
        z_q = z + (z_q - z).detach()

        # rearrange to original shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            embedding_ids=closest_embedding_ids,
        )

        return z_q, result_dict
