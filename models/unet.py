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
        skip = self.conv(x)   # save before pooling for the skip connection
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """Bilinear upsampling → concatenate skip → ConvBlock."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # After concatenation the channel count is in_ch + skip_ch
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # Upsample to match the spatial size of the skip-connection feature map
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    Lightweight U-Net for moiré removal.

    Architecture
    ------------
    Encoder (4 stages):  3 → 32 → 64 → 128 → 256
    Bottleneck:          256 → 512
    Decoder (4 stages):  512+256→256 → 256+128→128 → 128+64→64 → 64+32→32
    Head:                32 → 3 (Sigmoid)

    Parameter count ≈ 7 M — fits comfortably in 16 GB VRAM at batch size 8.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_ch: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, base_ch)        # 256 → 128
        self.enc2 = DownBlock(base_ch,     base_ch * 2)    # 128 → 64
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)    # 64  → 32
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)    # 32  → 16

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)  # 16 spatial

        # Decoder
        self.dec4 = UpBlock(base_ch * 16, base_ch * 8,  base_ch * 8)
        self.dec3 = UpBlock(base_ch * 8,  base_ch * 4,  base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4,  base_ch * 2,  base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2,  base_ch,      base_ch)

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, out_channels, kernel_size=1),
            nn.Sigmoid(),   # keep pixel values in [0, 1]
        )

    def forward(self, x):
        # --- Encoder ---
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Decoder (with skip connections) ---
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.head(x)


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
