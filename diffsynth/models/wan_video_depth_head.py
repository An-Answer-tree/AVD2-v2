import torch
import torch.nn as nn
from typing import Optional


class WanDepthHead(nn.Module):
    """A projection head for predicting depth latents from DiT hidden states.

    This module attaches to the output of the WanVideo Transformer backbone.
    It projects the high-dimensional hidden states down to the dimension
    required by the VAE latents (specifically for depth maps).

    Attributes:
        norm: A LayerNorm layer to normalize input features.
        proj: A Linear layer to project features to the target output dimension.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False
    ):
        """Initializes the WanDepthHead.

        Args:
            in_dim: The dimension of the input features (DiT hidden size).
            out_dim: The dimension of the output latents (VAE latent size).
            eps: Epsilon value for LayerNorm stability.
            elementwise_affine: Whether LayerNorm has learnable affine parameters.
                Defaults to False to maintain training stability, consistent
                with common DiT head designs.
        """
        super().__init__()

        # Normalization layer before projection.
        # We use elementwise_affine=False by default for stability in
        # post-backbone heads, but this can be tuned.
        self.norm = nn.LayerNorm(in_dim, eps=eps, elementwise_affine=elementwise_affine)

        # Linear projection to target dimension (e.g., 1536 -> 16).
        self.proj = nn.Linear(in_dim, out_dim)

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        """Initializes the weights of the projection layer.

        We use 'Zero Initialization' for the final projection. This ensures
        that at the start of training, the head outputs values close to zero,
        minimizing the initial loss shock and preventing the destruction of
        learned features in the pre-trained backbone.
        """
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of the depth head.

        Args:
            x: Input tensor of shape (Batch, Sequence_Length, Hidden_Dim).
               This represents the hidden states from the DiT backbone.
            t: Optional timestep tensor. Not used in this simple head but
               kept for API compatibility with the main video head.

        Returns:
            A Tensor of shape (Batch, Sequence_Length, Output_Dim).
        """
        # Apply normalization
        x = self.norm(x)

        # Project to latent dimension
        x = self.proj(x)

        return x