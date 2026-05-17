import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two Conv → BatchNorm → ReLU layers (the basic U-Net encoder block)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """ConvBlock followed by 2×2 max-pool (halves spatial dimensions)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class AttentionGate(nn.Module):
    """
    Soft attention gate applied to encoder skip connections.

    g = gating signal (decoder feature, lower resolution)
    x = skip connection (encoder feature, higher resolution)
    attention = sigmoid(W_psi(ReLU(W_g(g) + W_x(x) + b)))
    output    = attention * x
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample gating signal to match skip connection spatial resolution
        g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        attn = self.psi(self.relu(g1 + x1))
        return x * attn


class AttentionUpBlock(nn.Module):
    """Attention gate on skip → bilinear upsample → concatenate → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, F_int: int):
        super().__init__()
        self.attn = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=F_int)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        attn_skip = self.attn(x, skip)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, attn_skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Attention U-Net for moiré removal.

    Architecture
    ------------
    Encoder (4 stages):  3 → 64 → 128 → 256 → 512
    Bottleneck:          512 → 512
    Decoder (4 stages, each with an Attention Gate on the skip connection):
                         512+512→256 → 256+256→128 → 128+128→64 → 64+64→32
    Head:                32 → 3 (Tanh, residual added to input and clamped)

    Parameter count ≈ 13.7 M.
    Input/output: 3-channel RGB; spatial dimensions must be multiples of 16.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, base_ch)           # skip: base_ch   (64)
        self.enc2 = DownBlock(base_ch,     base_ch * 2)        # skip: base_ch*2 (128)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)        # skip: base_ch*4 (256)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)        # skip: base_ch*8 (512)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        # Decoder — each level applies an Attention Gate then ConvBlock
        self.dec4 = AttentionUpBlock(base_ch * 8, base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec3 = AttentionUpBlock(base_ch * 4, base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec2 = AttentionUpBlock(base_ch * 2, base_ch * 2, base_ch,     base_ch)
        self.dec1 = AttentionUpBlock(base_ch,     base_ch,     base_ch // 2, base_ch // 2)

        # Output head — produces a residual (can be negative), added to input
        self.head = nn.Sequential(
            nn.Conv2d(base_ch // 2, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        moire_input = x  # keep original for residual addition

        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (attention gates applied inside AttentionUpBlock)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        # Residual learning: predict noise to subtract, not the clean image
        residual = self.head(x)
        return torch.clamp(moire_input + residual - 0.5, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNet()
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.2f} M")
