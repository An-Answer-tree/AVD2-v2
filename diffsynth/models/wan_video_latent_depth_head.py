import torch
import torch.nn as nn
from typing import Optional


class WanLatentDepthHead(nn.Module):
    """Lightweight MLP head that maps video latents to depth latents.

    The head is intentionally small and does not participate in the diffusion
    process. It can be trained separately with frozen generators.

    Input/Output tensors are (B, C, T, H, W) in the same latent space as Wan VAE.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        out_channels: Optional[int] = None,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        out_channels = int(in_channels if out_channels is None else out_channels)
        self.in_channels = int(in_channels)
        self.out_channels = out_channels
        self.hidden_channels = int(hidden_channels)

        self.net = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
        )

        if zero_init:
            last = self.net[-1]
            if isinstance(last, nn.Conv3d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected (B,C,T,H,W) input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected C={self.in_channels}, got C={x.shape[1]}")
        return self.net(x)
