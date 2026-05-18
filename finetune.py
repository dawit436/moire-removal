"""
finetune.py — Fine-tune MBCNN on the moire-clean-dataset3 dataset.

Usage (Kaggle T4 or local):
    python finetune.py

Differences from train.py:
  - Uses MBCNN (models/mbcnn.py) instead of UNet
  - LR = 5e-5  (smaller for fine-tuning)
  - 30 epochs   (fine-tuning converges faster)
  - Loads pretrained MBCNN weights from PRETRAINED_CKPT if the file exists;
    otherwise trains from scratch
  - Best checkpoint auto-saved to /kaggle/working/best_mbcnn.pth
"""

import copy
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pytorch_msssim import ssim as compute_ssim
from tqdm import tqdm

from dataset import MoireDataset
from models.mbcnn import MBCNN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = (
    Path("/kaggle/input/datasets/dawitesubalew/moire-clean-dataset3/data")
    if Path("/kaggle").exists()
    else PROJECT_ROOT / "data"
)

CKPT_DIR = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# If this file exists its weights are loaded before training begins.
PRETRAINED_CKPT = CKPT_DIR / "best_mbcnn.pth"

BATCH_SIZE    = 4
EPOCHS        = 30
LR            = 5e-5
WARMUP_EPOCHS = 3
EMA_DECAY     = 0.999
CROP_SIZE     = 512
VAL_FRACTION  = 0.1
SAVE_EVERY    = 5
L1_WEIGHT     = 0.50
SSIM_WEIGHT   = 0.20
FFT_WEIGHT    = 0.30
SEED          = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def batch_psnr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([
        compute_psnr(preds[i], targets[i]) for i in range(preds.shape[0])
    ]))


def batch_ssim(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return compute_ssim(preds, targets, data_range=1.0, size_average=True).item()


# ---------------------------------------------------------------------------
# Loss  (identical to train.py: 0.50×L1 + 0.20×SSIM + 0.30×FFT)
# ---------------------------------------------------------------------------

def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag   = torch.log1p(torch.abs(torch.fft.fft2(pred)))
    target_mag = torch.log1p(torch.abs(torch.fft.fft2(target)))
    return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """0.50 × L1  +  0.20 × (1 − SSIM)  +  0.30 × FFT"""

    def __init__(self, l1_w=L1_WEIGHT, ssim_w=SSIM_WEIGHT, fft_w=FFT_WEIGHT):
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
# EMA  (identical to train.py)
# ---------------------------------------------------------------------------

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)


# ---------------------------------------------------------------------------
# Train / validate loops  (identical to train.py)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, ema=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        moire = batch["moire"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        pred = model(moire)
        loss = criterion(pred, clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)

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
    model = MBCNN().to(device)

    # Load pretrained weights if a checkpoint exists
    start_epoch = 1
    if PRETRAINED_CKPT.exists():
        print(f"\nLoading pretrained weights from {PRETRAINED_CKPT.name} ...")
        ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
        state_key = "ema_state" if "ema_state" in ckpt else "model_state"
        model.load_state_dict(ckpt[state_key], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch - 1}  "
              f"(best PSNR was {ckpt.get('val_psnr', float('nan')):.2f} dB)")
    else:
        print("\nNo pretrained checkpoint found — training MBCNN from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : MBCNN  ({total_params / 1e6:.2f} M parameters)")

    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    ema       = ModelEMA(model, decay=EMA_DECAY)

    warmup_sched = LinearLR(
        optimizer,
        start_factor=1e-6 / LR,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS - WARMUP_EPOCHS,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS],
    )

    print(f"LR     : {LR}  (warmup {WARMUP_EPOCHS} epochs → cosine decay)")
    print(f"EMA    : decay={EMA_DECAY}")

    # ----- Training loop ---------------------------------------------------
    best_psnr      = -float("inf")
    best_ckpt_path = CKPT_DIR / "best_mbcnn.pth"

    print(f"\nFine-tuning for {EPOCHS} epochs...\n{'=' * 60}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, ema)
        val_loss, val_psnr, val_ssim = validate(ema.model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PSNR (EMA): {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if epoch % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"mbcnn_epoch_{epoch:03d}.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "ema_state":   ema.model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_psnr":    val_psnr,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path.name}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "ema_state":   ema.model.state_dict(),
                "val_psnr":    best_psnr,
            }, best_ckpt_path)
            print(f"  ★ New best EMA PSNR {best_psnr:.2f} dB — saved best_mbcnn.pth")

            # Mirror to Kaggle output immediately so it survives a session crash
            kaggle_out = Path("/kaggle/working/best_mbcnn.pth")
            if Path("/kaggle").exists():
                shutil.copy(best_ckpt_path, kaggle_out)
                print(f"  → Also saved to {kaggle_out}")

    print(f"\nFine-tuning complete. Best EMA PSNR: {best_psnr:.2f} dB")

    # ----- Final save + download link (Kaggle) -----------------------------
    kaggle_out = Path("/kaggle/working/best_mbcnn.pth")
    if Path("/kaggle").exists():
        shutil.copy(best_ckpt_path, kaggle_out)
        print(f"\nModel saved to Kaggle output: {kaggle_out}")
        try:
            from IPython.display import FileLink, display
            display(FileLink("/kaggle/working/best_mbcnn.pth"))
            print("Click the link above to download the MBCNN model.")
        except Exception:
            print("Run this cell to get a download link:")
            print("  from IPython.display import FileLink")
            print("  FileLink('/kaggle/working/best_mbcnn.pth')")


if __name__ == "__main__":
    main()
