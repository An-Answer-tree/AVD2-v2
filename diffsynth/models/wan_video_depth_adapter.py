import torch


class WanDepthToVideoAdapter(torch.nn.Module):
    """Project depth latents into the video latent space.

    This adapter is designed to be lightweight and stable. It is initialized to
    output zeros so that enabling it does not change the base model outputs at
    the beginning of training.

    Input/Output tensors are expected to be (B, C, T, H, W).
    """

    def __init__(self, in_channels: int, out_channels: int, zero_init: bool = True):
        super().__init__()
        self.proj = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        if zero_init:
            torch.nn.init.zeros_(self.proj.weight)
            if self.proj.bias is not None:
                torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
