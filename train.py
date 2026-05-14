"""
train.py — Train the U-Net for moiré removal.

Usage (local or Kaggle):
    python train.py

All paths are relative to the directory where this script lives.
"""

import os
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_msssim import ssim as compute_ssim
from tqdm import tqdm

from dataset import MoireDataset
from models.unet import UNet


# ---------------------------------------------------------------------------
# Configuration — change these to tune the run
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).parent          # directory of this script
DATA_DIR      = (
    Path("/kaggle/input/datasets/dawitesubalew/moire-pattern-dataset2/data")
    if Path("/kaggle").exists()
    else PROJECT_ROOT / "data"
)
CKPT_DIR      = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

BATCH_SIZE    = 4
EPOCHS        = 50
LR            = 1e-4
CROP_SIZE     = 512
VAL_FRACTION  = 0.1   # fraction of training data used for validation
SAVE_EVERY    = 5     # save a checkpoint every N epochs
L1_WEIGHT     = 0.70
SSIM_WEIGHT   = 0.15
FFT_WEIGHT    = 0.15  # weights sum to 1.0
SEED          = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio (dB). Both tensors in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def batch_psnr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Average PSNR over a batch."""
    return float(np.mean([
        compute_psnr(preds[i], targets[i]) for i in range(preds.shape[0])
    ]))


def batch_ssim(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Average SSIM over a batch (uses pytorch-msssim)."""
    return compute_ssim(preds, targets, data_range=1.0, size_average=True).item()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on log-scaled FFT magnitudes — balances low and high frequencies."""
    pred_mag   = torch.log1p(torch.abs(torch.fft.fft2(pred)))
    target_mag = torch.log1p(torch.abs(torch.fft.fft2(target)))
    return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """0.70 × L1  +  0.15 × (1 − SSIM)  +  0.15 × FFT"""

    def __init__(
        self,
        l1_w:   float = L1_WEIGHT,
        ssim_w: float = SSIM_WEIGHT,
        fft_w:  float = FFT_WEIGHT,
    ):
        super().__init__()
        self.l1_w   = l1_w
        self.ssim_w = ssim_w
        self.fft_w  = fft_w
        self.l1     = nn.L1Loss()

    def forward(self, pred, target):
        l1   = self.l1(pred, target)
        ssim = 1.0 - compute_ssim(pred, target, data_range=1.0, size_average=True)
        fft  = fft_loss(pred, target)
        return self.l1_w * l1 + self.ssim_w * ssim + self.fft_w * fft


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        moire = batch["moire"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        pred = model(moire)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_psnr, total_ssim = 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        moire = batch["moire"].to(device)
        clean = batch["clean"].to(device)

        pred = model(moire)
        loss = criterion(pred, clean)

        total_loss += loss.item()
        total_psnr += batch_psnr(pred.cpu(), clean.cpu())
        total_ssim += batch_ssim(pred.cpu(), clean.cpu())

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ----- Dataset ---------------------------------------------------------
    full_dataset = MoireDataset(DATA_DIR, split="train", crop_size=CROP_SIZE)

    val_size   = max(1, int(len(full_dataset) * VAL_FRACTION))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Dataset: {train_size} train / {val_size} val samples")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=1, pin_memory=device.type == "cuda",
    )

    # ----- Model -----------------------------------------------------------
    model     = UNet().to(device)
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model : {total_params / 1e6:.2f} M parameters")

    # ----- Training --------------------------------------------------------
    best_psnr      = -float("inf")
    best_ckpt_path = CKPT_DIR / "best_model.pth"

    print(f"\nStarting training for {EPOCHS} epochs...\n{'=' * 60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"epoch_{epoch:03d}.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_psnr":    val_psnr,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path.name}")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_psnr":    best_psnr,
            }, best_ckpt_path)
            print(f"  ★ New best PSNR {best_psnr:.2f} dB — saved best_model.pth")
            kaggle_out = Path("/kaggle/working/best_model_attention.pth")
            if Path("/kaggle").exists():
                shutil.copy(best_ckpt_path, kaggle_out)
                print(f"  → Also saved to {kaggle_out}")

    print(f"\nTraining complete. Best validation PSNR: {best_psnr:.2f} dB")

    # Auto-save to Kaggle output and generate download link
    kaggle_out = Path("/kaggle/working/best_model_attention.pth")
    if Path("/kaggle").exists():
        shutil.copy(CKPT_DIR / "best_model.pth", kaggle_out)
        print(f"\nModel saved to: {kaggle_out}")

        try:
            from IPython.display import FileLink, display
            display(FileLink("/kaggle/working/best_model_attention.pth"))
            print("Click the link above to download the model.")
        except Exception:
            print("Run this to download:")
            print("from IPython.display import FileLink; FileLink('/kaggle/working/best_model_attention.pth')")


if __name__ == "__main__":
    main()
