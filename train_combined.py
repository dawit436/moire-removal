"""
train_combined.py — Train MBCNN on FHDMi, TIP2018, or both datasets.

Usage:
    python train_combined.py --dataset fhdmi
    python train_combined.py --dataset tip2018
    python train_combined.py --dataset combined
    python train_combined.py --dataset combined --pretrained checkpoints/best_mbcnn.pth

Optimised for RTX 2080 Ti (11 GB VRAM):
    BATCH_SIZE=8  |  CROP_SIZE=512  |  EPOCHS=100  |  LR=1e-4
"""

import argparse
import copy
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pytorch_msssim import ssim as compute_ssim
from tqdm import tqdm

from dataset import MoireDataset
from models.mbcnn import MBCNN


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
FHDMI_DATA    = Path("D:/FHDMi/data")
TIP2018_DATA  = Path("D:/TIP2018/data")
CKPT_DIR      = PROJECT_ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Hyper-parameters (RTX 2080 Ti, 11 GB VRAM) ────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 100
LR            = 1e-4
WARMUP_EPOCHS = 5
EMA_DECAY     = 0.999
CROP_SIZE     = 512
VAL_FRACTION  = 0.1
SAVE_EVERY    = 10
L1_WEIGHT     = 0.50
SSIM_WEIGHT   = 0.20
FFT_WEIGHT    = 0.30
SEED          = 42


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def batch_psnr(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return float(np.mean([compute_psnr(preds[i], targets[i]) for i in range(preds.shape[0])]))


def batch_ssim(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return compute_ssim(preds, targets, data_range=1.0, size_average=True).item()


# ── Loss ──────────────────────────────────────────────────────────────────────
def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag   = torch.log1p(torch.abs(torch.fft.fft2(pred)))
    target_mag = torch.log1p(torch.abs(torch.fft.fft2(target)))
    return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    """0.50 × L1  +  0.20 × (1 − SSIM)  +  0.30 × FFT-L1"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1   = self.l1(pred, target)
        ssim = 1.0 - compute_ssim(pred, target, data_range=1.0, size_average=True)
        fft  = fft_loss(pred, target)
        return L1_WEIGHT * l1 + SSIM_WEIGHT * ssim + FFT_WEIGHT * fft


# ── EMA ───────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
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


# ── Train / Validate ──────────────────────────────────────────────────────────
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
    total_loss = total_psnr = total_ssim = 0.0
    for batch in tqdm(loader, desc="  Val  ", leave=False):
        moire = batch["moire"].to(device)
        clean = batch["clean"].to(device)
        pred  = model(moire)
        total_loss += criterion(pred, clean).item()
        total_psnr += batch_psnr(pred.cpu(), clean.cpu())
        total_ssim += batch_ssim(pred.cpu(), clean.cpu())
    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


# ── Dataset loading ───────────────────────────────────────────────────────────
def _split_dataset(ds):
    val_size   = max(1, int(len(ds) * VAL_FRACTION))
    train_size = len(ds) - val_size
    return random_split(ds, [train_size, val_size],
                        generator=torch.Generator().manual_seed(SEED))


def load_datasets(name: str):
    """Return (train_dataset, val_dataset) for the requested --dataset."""
    if name == "fhdmi":
        if not FHDMI_DATA.exists():
            raise FileNotFoundError(
                f"FHDMi data not found at {FHDMI_DATA}. "
                "Run organize_all_datasets.py --fhdmi first."
            )
        ds = MoireDataset(FHDMI_DATA, split="train", crop_size=CROP_SIZE)
        print(f"  FHDMi train: {len(ds)} samples")
        return _split_dataset(ds)

    if name == "tip2018":
        if not TIP2018_DATA.exists():
            raise FileNotFoundError(
                f"TIP2018 data not found at {TIP2018_DATA}. "
                "Run organize_all_datasets.py --tip2018 first."
            )
        ds = MoireDataset(TIP2018_DATA, split="train", crop_size=CROP_SIZE)
        print(f"  TIP2018 train: {len(ds)} samples")
        return _split_dataset(ds)

    if name == "combined":
        parts = []
        if FHDMI_DATA.exists():
            ds = MoireDataset(FHDMI_DATA, split="train", crop_size=CROP_SIZE)
            print(f"  FHDMi   train: {len(ds):,} samples")
            parts.append(ds)
        else:
            print(f"  [WARN] FHDMi not found at {FHDMI_DATA} — skipping")

        if TIP2018_DATA.exists():
            ds = MoireDataset(TIP2018_DATA, split="train", crop_size=CROP_SIZE)
            print(f"  TIP2018 train: {len(ds):,} samples")
            parts.append(ds)
        else:
            print(f"  [WARN] TIP2018 not found at {TIP2018_DATA} — skipping")

        if not parts:
            raise FileNotFoundError(
                "No datasets found. Run organize_all_datasets.py first."
            )
        combined = ConcatDataset(parts)
        print(f"  Combined total: {len(combined):,} samples")
        return _split_dataset(combined)

    raise ValueError(f"Unknown dataset '{name}'. Choose: fhdmi | tip2018 | combined")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["fhdmi", "tip2018", "combined"],
        default="combined",
        help="Dataset to train on (default: combined)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (default: checkpoints/best_mbcnn.pth)",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Dataset : {args.dataset}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds = load_datasets(args.dataset)
    print(f"Split   : {len(train_ds):,} train / {len(val_ds):,} val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda",
        persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MBCNN().to(device)

    pretrained_path = (
        Path(args.pretrained) if args.pretrained
        else CKPT_DIR / "best_mbcnn.pth"
    )
    start_epoch = 1
    if pretrained_path.exists():
        print(f"\nLoading pretrained weights from {pretrained_path.name} ...")
        ckpt      = torch.load(pretrained_path, map_location=device)
        state_key = "ema_state" if "ema_state" in ckpt else "model_state"
        model.load_state_dict(ckpt[state_key], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from epoch {start_epoch - 1}  "
              f"(best PSNR: {ckpt.get('val_psnr', float('nan')):.2f} dB)")
    else:
        print("\nNo pretrained checkpoint — training MBCNN from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : MBCNN ({total_params / 1e6:.2f} M parameters)")

    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    ema       = ModelEMA(model, decay=EMA_DECAY)

    warmup_sched = LinearLR(
        optimizer, start_factor=1e-6 / LR, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )

    print(f"LR      : {LR}  (warmup {WARMUP_EPOCHS} epochs → cosine decay to 1e-6)")
    print(f"Batch   : {BATCH_SIZE}  |  Crop: {CROP_SIZE}  |  Epochs: {EPOCHS}")
    print(f"EMA     : decay={EMA_DECAY}  |  Grad-clip: max_norm=1.0")
    print(f"Loss    : {L1_WEIGHT}×L1 + {SSIM_WEIGHT}×(1-SSIM) + {FFT_WEIGHT}×FFT")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_psnr      = -float("inf")
    best_ckpt_name = f"best_mbcnn_{args.dataset}.pth"
    best_ckpt_path = CKPT_DIR / best_ckpt_name

    print(f"\nTraining for {EPOCHS} epochs...\n{'=' * 70}")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, ema
        )
        val_loss, val_psnr, val_ssim = validate(
            ema.model, val_loader, criterion, device
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        if epoch % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"mbcnn_{args.dataset}_epoch_{epoch:03d}.pth"
            torch.save({
                "epoch":       epoch,
                "dataset":     args.dataset,
                "model_state": model.state_dict(),
                "ema_state":   ema.model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_psnr":    val_psnr,
            }, ckpt_path)
            print(f"  → Checkpoint: {ckpt_path.name}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch":       epoch,
                "dataset":     args.dataset,
                "model_state": model.state_dict(),
                "ema_state":   ema.model.state_dict(),
                "val_psnr":    best_psnr,
            }, best_ckpt_path)
            print(f"  ★ New best PSNR {best_psnr:.2f} dB — saved {best_ckpt_name}")

            if Path("/kaggle").exists():
                shutil.copy(best_ckpt_path, Path("/kaggle/working") / best_ckpt_name)

    print(f"\nTraining complete. Best EMA PSNR: {best_psnr:.2f} dB")
    print(f"Best model: {best_ckpt_path}")

    # ── Download link (Kaggle) ────────────────────────────────────────────────
    if Path("/kaggle").exists():
        kaggle_out = Path("/kaggle/working") / best_ckpt_name
        shutil.copy(best_ckpt_path, kaggle_out)
        print(f"\nModel saved to Kaggle output: {kaggle_out}")
        try:
            from IPython.display import FileLink, display
            display(FileLink(str(kaggle_out)))
            print("Click the link above to download the model.")
        except Exception:
            print(f"  from IPython.display import FileLink")
            print(f"  FileLink('{kaggle_out}')")


if __name__ == "__main__":
    main()
