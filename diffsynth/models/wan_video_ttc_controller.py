#Added by Cheng Li
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .wan_video_dit import sinusoidal_embedding_1d


class WanTTCTokenizer(nn.Module):
    def __init__(
        self,
        out_dim: int,
        freq_dim: int = 256,
        hidden_dim: int = 512,
        max_ttc: float = 512.0,
        add_pos_emb: bool = True,
        init_zero_last: bool = True,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.freq_dim = int(freq_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_ttc = float(max_ttc) if max_ttc is not None else None
        self.add_pos_emb = bool(add_pos_emb)

        self.value_mlp = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.pos_mlp = (
            nn.Sequential(
                nn.Linear(self.freq_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
            if self.add_pos_emb
            else None
        )

        if init_zero_last:
            self._init_zero_last_layers()

    def _init_zero_last_layers(self):
        mlps = [self.value_mlp]
        if self.pos_mlp is not None:
            mlps.append(self.pos_mlp)
        for mlp in mlps:
            last = mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)

    def forward(
        self,
        ttc: torch.Tensor,
        num_frames: Optional[int] = None,
        downsample_factor: int = 4,
    ) -> torch.Tensor:
        if not isinstance(ttc, torch.Tensor):
            ttc = torch.as_tensor(ttc)

        if ttc.dim() == 0:
            ttc = ttc.view(1, 1)
        elif ttc.dim() == 1:
            ttc = ttc.unsqueeze(0)
        elif ttc.dim() != 2:
            raise ValueError(f"ttc must be 0D/1D/2D tensor, got shape={tuple(ttc.shape)}")

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        ttc = ttc.to(device=device, dtype=torch.float32)
        bsz, t_len = ttc.shape

        if num_frames is not None:
            if isinstance(num_frames, torch.Tensor):
                num_frames = int(num_frames.item())
            num_frames = int(num_frames)

            if t_len >= num_frames:
                ttc = ttc[:, :num_frames]
            else:
                pad = ttc[:, -1:].repeat(1, num_frames - t_len)
                ttc = torch.cat([ttc, pad], dim=1)

        if num_frames is not None and downsample_factor is not None and int(downsample_factor) > 1:
            downsample_factor = int(downsample_factor)
            idx = torch.arange(0, num_frames, downsample_factor, device=device)
            ttc = ttc.index_select(dim=1, index=idx)

        if self.max_ttc is not None:
            ttc = ttc.clamp(min=-self.max_ttc, max=self.max_ttc)

        l_tok = ttc.shape[1]

        emb_val = sinusoidal_embedding_1d(self.freq_dim, ttc.reshape(-1)).to(device=device, dtype=dtype)
        emb_val = emb_val.view(bsz, l_tok, self.freq_dim)
        out = self.value_mlp(emb_val)

        if self.pos_mlp is not None:
            pos = torch.arange(l_tok, device=device, dtype=torch.float32)
            emb_pos = sinusoidal_embedding_1d(self.freq_dim, pos).to(device=device, dtype=dtype)
            emb_pos = emb_pos.unsqueeze(0).expand(bsz, -1, -1)
            out = out + self.pos_mlp(emb_pos)

        return out
