"""
models/mbcnn.py — Multi-Branch CNN for moire pattern removal.

Inspired by: "Moire Pattern Removal via Attentive Residual Feature Composition"
(Zhenng et al., 2021 — Learnable Bandpass Filter / MBCNN)

Architecture
------------
1. Learned frequency decomposition (depthwise 7×7 blur):
       input → low / mid / high components
2. Three parallel residual branches (one per frequency band)
3. Channel-attention-weighted fusion → residual output

~5 M parameters. forward(x) API matches UNet in models/unet.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Conv-BN-ReLU → Conv-BN with identity skip, followed by ReLU."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class FrequencyBranch(nn.Module):
    """3-ch input → project to branch_ch → N residual blocks."""

    def __init__(self, in_ch: int, branch_ch: int, n_blocks: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(branch_ch) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class LearnedDecomposition(nn.Module):
    """
    Decomposes the RGB input into three frequency bands.

    low  = learned depthwise blur (7×7, init = uniform average)
    high = input − low   (fine detail / moire residual)
    mid  = 0.5 × (input + low)

    The blur kernel is unconstrained after init so the network can
    adapt the decomposition to the moire frequency range during training.
    """

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.blur = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3,
                              groups=in_ch, bias=False)
        nn.init.constant_(self.blur.weight, 1.0 / 49.0)

    def forward(self, x: torch.Tensor):
        low  = self.blur(x)
        high = x - low
        mid  = 0.5 * (x + low)
        return low, mid, high


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (global avg-pool → FC → sigmoid)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


# ---------------------------------------------------------------------------
# MBCNN
# ---------------------------------------------------------------------------

class MBCNN(nn.Module):
    """
    Multi-Branch CNN for moire removal.

    Input/output: 3-channel RGB float32 in [0, 1], arbitrary spatial size.
    Default config: branch_ch=128, n_blocks=5 → ~5.0 M parameters.

    Residual connection: output = clamp(input + head(fused) − 0.5, 0, 1)
    This mirrors the UNet residual convention so both models can be compared
    under identical training and evaluation code.
    """

    def __init__(
        self,
        in_channels:  int = 3,
        out_channels: int = 3,
        branch_ch:    int = 128,
        n_blocks:     int = 5,
    ):
        super().__init__()

        self.decompose = LearnedDecomposition(in_channels)

        self.low_branch  = FrequencyBranch(in_channels, branch_ch, n_blocks)
        self.mid_branch  = FrequencyBranch(in_channels, branch_ch, n_blocks)
        self.high_branch = FrequencyBranch(in_channels, branch_ch, n_blocks)

        fused_ch = branch_ch * 3
        self.attention = ChannelAttention(fused_ch, reduction=8)

        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, branch_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moire_input = x

        low, mid, high = self.decompose(x)

        f_low  = self.low_branch(low)
        f_mid  = self.mid_branch(mid)
        f_high = self.high_branch(high)

        fused = torch.cat([f_low, f_mid, f_high], dim=1)
        fused = self.attention(fused)
        residual = self.fusion(fused)

        return torch.clamp(moire_input + residual - 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = MBCNN()
    dummy = torch.randn(2, 3, 256, 256)
    out   = model(dummy)
    print(f"Input      : {dummy.shape}")
    print(f"Output     : {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total / 1e6:.2f} M")
