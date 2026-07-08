"""
train_combined.py — Train MBCNN on FHDMi, TIP2018, or both datasets.

Usage:
    python train_combined.py --dataset fhdmi
    python train_combined.py --dataset tip2018
    python train_combined.py --dataset combined
    python train_combined.py --dataset combined --pretrained checkpoints/best_mbcnn.pth

Optimised for RTX 2080 Ti (11 GB VRAM):
    BATCH_SIZE=8  |  CROP_SIZE=512  |  EPOCHS=30  |  LR=1e-4
"""

import argparse
import copy
from contextlib import nullcontext
import json
import math
import os
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
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim as compute_ssim
from tqdm import tqdm
from PIL import Image, ImageDraw

from dataset import MoireDataset
from models.mbcnn import MBCNN


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
KAGGLE_ROOT = Path("/kaggle")

DEFAULT_CKPT_DIR = (
    Path("/kaggle/working/checkpoints")
    if KAGGLE_ROOT.exists()
    else PROJECT_ROOT / "checkpoints"
)

FHDMI_DATA = Path(os.environ.get("FHDMI_DATA_ROOT", "D:/FHDMi/data"))
TIP2018_DATA = Path(os.environ.get("TIP2018_DATA_ROOT", "D:/TIP2018/data"))
CKPT_DIR = Path(os.environ.get("MBCNN_CKPT_DIR", str(DEFAULT_CKPT_DIR)))
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters (RTX 2080 Ti, 11 GB VRAM) ────────────────────────────────
BATCH_SIZE    = 8
EPOCHS        = 30
LR            = 1e-4
WARMUP_EPOCHS = 5
EMA_DECAY     = 0.999
CROP_SIZE     = 512
VAL_FRACTION  = 0.1
SAVE_EVERY    = 10
NUM_WORKERS   = 2
ACCUM_STEPS   = 1
SCALE_JITTER  = False
HARD_CROP_CANDIDATES = 1
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


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def build_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


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
def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    ema=None,
    scaler=None,
    use_amp=False,
    accum_steps=1,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False), start=1):
        moire = batch["moire"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            pred = model(moire)

        # Keep FFT/SSIM loss in float32; mixed precision here can become unstable.
        loss = criterion(pred.float(), clean.float())
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at step {step}: {loss.item()}"
            )
        backward_loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        if step % accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    raise RuntimeError(
                        f"Non-finite gradient norm at step {step}: {grad_norm.item()}"
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(
                        f"Non-finite gradient norm at step {step}: {grad_norm.item()}"
                    )
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    total_input_psnr = total_input_ssim = 0.0
    for batch in tqdm(loader, desc="  Val  ", leave=False):
        moire = batch["moire"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        pred  = model(moire)
        total_loss += criterion(pred, clean).item()
        total_psnr += batch_psnr(pred.cpu(), clean.cpu())
        total_ssim += batch_ssim(pred.cpu(), clean.cpu())
        total_input_psnr += batch_psnr(moire.cpu(), clean.cpu())
        total_input_ssim += batch_ssim(moire.cpu(), clean.cpu())
    n = len(loader)
    return (
        total_loss / n,
        total_psnr / n,
        total_ssim / n,
        total_input_psnr / n,
        total_input_ssim / n,
    )


def _label_image(image: Image.Image, label: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 26), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255))
    return out


@torch.no_grad()
def save_validation_preview(model, loader, device, out_path: Path, max_items: int = 3):
    model.eval()
    rows = []
    for batch in loader:
        moire = batch["moire"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        pred = model(moire).clamp(0, 1).cpu()

        for i in range(moire.shape[0]):
            panels = [
                ("input", TF.to_pil_image(moire[i].cpu().clamp(0, 1))),
                ("output", TF.to_pil_image(pred[i])),
                ("gt", TF.to_pil_image(clean[i].cpu().clamp(0, 1))),
            ]
            labeled = [_label_image(img, label) for label, img in panels]
            row = Image.new(
                "RGB",
                (sum(img.width for img in labeled), max(img.height for img in labeled)),
                color=(245, 245, 245),
            )
            x = 0
            for img in labeled:
                row.paste(img, (x, 0))
                x += img.width
            rows.append(row)
            if len(rows) >= max_items:
                break
        if len(rows) >= max_items:
            break

    if not rows:
        return

    grid = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)))
    y = 0
    for row in rows:
        grid.paste(row, (0, y))
        y += row.height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=95)


# ── Dataset loading ───────────────────────────────────────────────────────────
def _split_dataset(ds):
    val_size   = max(1, int(len(ds) * VAL_FRACTION))
    train_size = len(ds) - val_size
    return random_split(ds, [train_size, val_size],
                        generator=torch.Generator().manual_seed(SEED))


def _has_split(root: Path, split: str) -> bool:
    return (root / split / "moire").exists() and (root / split / "clean").exists()


def _single_dataset_train_val(root: Path, label: str):
    train_ds = MoireDataset(
        root,
        split="train",
        crop_size=CROP_SIZE,
        scale_jitter=SCALE_JITTER,
        hard_crop_candidates=HARD_CROP_CANDIDATES,
    )
    print(f"  {label:<7} train: {len(train_ds):,} samples")

    if _has_split(root, "test"):
        val_ds = MoireDataset(root, split="test", crop_size=CROP_SIZE)
        print(f"  {label:<7} val  : {len(val_ds):,} samples (test split)")
        return train_ds, val_ds

    print(f"  [WARN] {label} has no test split - using {VAL_FRACTION:.0%} train holdout")
    return _split_dataset(train_ds)


def _maybe_concat(parts):
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def load_datasets(name: str):
    """Return (train_dataset, val_dataset) for the requested --dataset."""
    if name == "fhdmi":
        if not FHDMI_DATA.exists():
            raise FileNotFoundError(
                f"FHDMi data not found at {FHDMI_DATA}. "
                "Pass --fhdmi-data or set FHDMI_DATA_ROOT."
            )
        return _single_dataset_train_val(FHDMI_DATA, "FHDMi")

    if name == "tip2018":
        if not TIP2018_DATA.exists():
            raise FileNotFoundError(
                f"TIP2018 data not found at {TIP2018_DATA}. "
                "Pass --tip2018-data or set TIP2018_DATA_ROOT."
            )
        return _single_dataset_train_val(TIP2018_DATA, "TIP2018")

    if name == "combined":
        train_parts = []
        val_parts = []
        if FHDMI_DATA.exists():
            train_ds, val_ds = _single_dataset_train_val(FHDMI_DATA, "FHDMi")
            train_parts.append(train_ds)
            val_parts.append(val_ds)
        else:
            print(f"  [WARN] FHDMi not found at {FHDMI_DATA} — skipping")

        if TIP2018_DATA.exists():
            train_ds, val_ds = _single_dataset_train_val(TIP2018_DATA, "TIP2018")
            train_parts.append(train_ds)
            val_parts.append(val_ds)
        else:
            print(f"  [WARN] TIP2018 not found at {TIP2018_DATA} — skipping")

        if not train_parts:
            raise FileNotFoundError(
                "No datasets found. Pass --fhdmi-data/--tip2018-data or set env vars."
            )
        train_ds = _maybe_concat(train_parts)
        val_ds = _maybe_concat(val_parts)
        print(f"  Combined train: {len(train_ds):,} samples")
        print(f"  Combined val  : {len(val_ds):,} samples")
        return train_ds, val_ds

    raise ValueError(f"Unknown dataset '{name}'. Choose: fhdmi | tip2018 | combined")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global FHDMI_DATA, TIP2018_DATA, CKPT_DIR
    global BATCH_SIZE, EPOCHS, LR, CROP_SIZE, VAL_FRACTION, SAVE_EVERY
    global NUM_WORKERS, ACCUM_STEPS, SCALE_JITTER, HARD_CROP_CANDIDATES
    global L1_WEIGHT, SSIM_WEIGHT, FFT_WEIGHT

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
        help="Path to weights/checkpoint for fine-tuning. Use 'none' to disable auto-load.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume epoch and optimizer state from --pretrained.")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming/fine-tuning, keep weights but start with a fresh optimizer/scheduler.",
    )
    parser.add_argument("--fhdmi-data", type=Path, default=None, help="FHDMi data root containing train/ and test/.")
    parser.add_argument("--tip2018-data", type=Path, default=None, help="TIP2018 data root containing train/ and test/.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Checkpoint/output directory.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--scale-jitter", action="store_true", default=SCALE_JITTER)
    parser.add_argument("--hard-crop-candidates", type=int, default=HARD_CROP_CANDIDATES)
    parser.add_argument("--l1-weight", type=float, default=L1_WEIGHT)
    parser.add_argument("--ssim-weight", type=float, default=SSIM_WEIGHT)
    parser.add_argument("--fft-weight", type=float, default=FFT_WEIGHT)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    args = parser.parse_args()

    if args.fhdmi_data is not None:
        FHDMI_DATA = args.fhdmi_data
    if args.tip2018_data is not None:
        TIP2018_DATA = args.tip2018_data
    if args.out_dir is not None:
        CKPT_DIR = args.out_dir

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    CROP_SIZE = args.crop_size
    VAL_FRACTION = args.val_fraction
    SAVE_EVERY = args.save_every
    NUM_WORKERS = args.num_workers
    ACCUM_STEPS = max(1, args.accum_steps)
    SCALE_JITTER = args.scale_jitter
    HARD_CROP_CANDIDATES = max(1, args.hard_crop_candidates)
    L1_WEIGHT = args.l1_weight
    SSIM_WEIGHT = args.ssim_weight
    FFT_WEIGHT = args.fft_weight
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Dataset : {args.dataset}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds = load_datasets(args.dataset)
    print(f"Split   : {len(train_ds):,} train / {len(val_ds):,} val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MBCNN().to(device)

    if args.pretrained and args.pretrained.lower() == "none":
        pretrained_path = None
    else:
        pretrained_path = (
            Path(args.pretrained) if args.pretrained
            else CKPT_DIR / "best_mbcnn.pth"
        )
    start_epoch = 1
    loaded_ckpt = None
    if pretrained_path is not None and pretrained_path.exists():
        print(f"\nLoading pretrained weights from {pretrained_path.name} ...")
        loaded_ckpt = torch.load(pretrained_path, map_location=device)
        if args.resume and isinstance(loaded_ckpt, dict) and "model_state" in loaded_ckpt:
            state_dict = loaded_ckpt["model_state"]
        elif isinstance(loaded_ckpt, dict) and "ema_state" in loaded_ckpt:
            state_dict = loaded_ckpt["ema_state"]
        elif isinstance(loaded_ckpt, dict) and "model_state" in loaded_ckpt:
            state_dict = loaded_ckpt["model_state"]
        else:
            state_dict = loaded_ckpt
        model.load_state_dict(state_dict, strict=False)
        if args.resume:
            start_epoch = loaded_ckpt.get("epoch", 0) + 1 if isinstance(loaded_ckpt, dict) else 1
            print(f"  Resuming from epoch {start_epoch - 1}  "
                  f"(best PSNR: {loaded_ckpt.get('val_psnr', float('nan')):.2f} dB)"
                  if isinstance(loaded_ckpt, dict) else "  Resuming from raw state_dict")
        else:
            print(f"  Loaded weights for fine-tuning "
                  f"(source epoch: {loaded_ckpt.get('epoch', 'unknown') if isinstance(loaded_ckpt, dict) else 'unknown'})")
    else:
        print("\nNo pretrained checkpoint — training MBCNN from scratch.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : MBCNN ({total_params / 1e6:.2f} M parameters)")

    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    ema       = ModelEMA(model, decay=EMA_DECAY)
    if args.resume and isinstance(loaded_ckpt, dict) and "ema_state" in loaded_ckpt:
        ema.model.load_state_dict(loaded_ckpt["ema_state"], strict=False)

    warmup_sched = LinearLR(
        optimizer, start_factor=1e-6 / LR, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )
    use_amp = args.amp and device.type == "cuda"
    scaler = build_grad_scaler(use_amp)

    if args.resume and isinstance(loaded_ckpt, dict):
        can_exact_resume = (
            "optim_state" in loaded_ckpt
            and "scheduler_state" in loaded_ckpt
            and not args.reset_optimizer
        )
        if can_exact_resume:
            optimizer.load_state_dict(loaded_ckpt["optim_state"])
            scheduler.load_state_dict(loaded_ckpt["scheduler_state"])
            if scaler is not None and "scaler_state" in loaded_ckpt:
                scaler.load_state_dict(loaded_ckpt["scaler_state"])
            print("  Exact resume: optimizer, scheduler and scaler state restored.")
        elif args.reset_optimizer:
            print("  Resume weights only: optimizer/scheduler reset by request.")
        elif "optim_state" in loaded_ckpt:
            print(
                "  Resume weights only: optimizer state exists but scheduler state is missing; "
                "resetting optimizer/scheduler to avoid LR mismatch."
            )
        else:
            print("  Resume weights only: checkpoint has no optimizer/scheduler state.")

    print(f"LR      : {LR}  (warmup {WARMUP_EPOCHS} epochs → cosine decay to 1e-6)")
    print(f"Batch   : {BATCH_SIZE}  |  Accum: {ACCUM_STEPS}  |  Crop: {CROP_SIZE}  |  Epochs: {EPOCHS}")
    print(
        f"Workers : {NUM_WORKERS}  |  AMP: {use_amp}  |  "
        f"Scale jitter: {SCALE_JITTER}  |  Hard crops: {HARD_CROP_CANDIDATES}"
    )
    print(f"Output  : {CKPT_DIR}")
    print(f"EMA     : decay={EMA_DECAY}  |  Grad-clip: max_norm=1.0")
    print(f"Loss    : {L1_WEIGHT}×L1 + {SSIM_WEIGHT}×(1-SSIM) + {FFT_WEIGHT}×FFT")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_epoch     = 0
    best_psnr      = -float("inf")
    best_ckpt_name = f"best_mbcnn_{args.dataset}.pth"
    best_ckpt_path = CKPT_DIR / best_ckpt_name
    last_ckpt_name = f"last_mbcnn_{args.dataset}.pth"
    last_ckpt_path = CKPT_DIR / last_ckpt_name
    summary_path = CKPT_DIR / "training_summary.json"

    def checkpoint_payload(
        epoch: int,
        val_psnr: float,
        val_ssim: float,
        input_psnr: float,
        input_ssim: float,
    ):
        payload = {
            "epoch":           epoch,
            "dataset":         args.dataset,
            "model_state":     model.state_dict(),
            "ema_state":       ema.model.state_dict(),
            "optim_state":     optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_psnr":        val_psnr,
            "val_ssim":        val_ssim,
            "input_psnr":      input_psnr,
            "input_ssim":      input_ssim,
            "delta_psnr":      val_psnr - input_psnr,
            "best_psnr":       best_psnr,
            "best_epoch":      best_epoch,
        }
        if scaler is not None:
            payload["scaler_state"] = scaler.state_dict()
        return payload

    if args.resume and isinstance(loaded_ckpt, dict):
        ckpt_best = loaded_ckpt.get("best_psnr", loaded_ckpt.get("val_psnr"))
        if ckpt_best is not None:
            best_psnr = float(ckpt_best)
            best_epoch = int(loaded_ckpt.get("best_epoch", loaded_ckpt.get("epoch", 0)))
            print(f"  Resume best baseline: epoch {best_epoch}, PSNR {best_psnr:.2f} dB")

    if start_epoch > EPOCHS:
        raise SystemExit(
            f"start_epoch={start_epoch} is greater than --epochs={EPOCHS}. "
            "Increase --epochs or fine-tune without --resume."
        )

    print(f"\nTraining for {EPOCHS} epochs...\n{'=' * 70}")

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            ema,
            scaler=scaler,
            use_amp=use_amp,
            accum_steps=ACCUM_STEPS,
        )
        val_loss, val_psnr, val_ssim, input_psnr, input_ssim = validate(
            ema.model, val_loader, criterion, device
        )
        delta_psnr = val_psnr - input_psnr
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f} | "
            f"Input PSNR: {input_psnr:.2f} dB | "
            f"Delta: {delta_psnr:+.2f} dB | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        is_new_best = val_psnr > best_psnr
        if is_new_best:
            best_epoch = epoch
            best_psnr = val_psnr

        if epoch % SAVE_EVERY == 0:
            ckpt_path = CKPT_DIR / f"mbcnn_{args.dataset}_epoch_{epoch:03d}.pth"
            torch.save(
                checkpoint_payload(epoch, val_psnr, val_ssim, input_psnr, input_ssim),
                ckpt_path,
            )
            print(f"  → Checkpoint: {ckpt_path.name}")

        if is_new_best:
            torch.save(
                checkpoint_payload(epoch, best_psnr, val_ssim, input_psnr, input_ssim),
                best_ckpt_path,
            )
            print(f"  ★ New best PSNR {best_psnr:.2f} dB — saved {best_ckpt_name}")

            preview_name = f"val_preview_{args.dataset}_epoch_{epoch:03d}.jpg"
            preview_path = CKPT_DIR / preview_name
            save_validation_preview(ema.model, val_loader, device, preview_path)
            print(f"  → Preview: {preview_name}")

            summary_path.write_text(json.dumps({
                "dataset": args.dataset,
                "completed": False,
                "best_epoch": epoch,
                "best_psnr": best_psnr,
                "best_ssim": val_ssim,
                "input_psnr": input_psnr,
                "input_ssim": input_ssim,
                "delta_psnr": delta_psnr,
                "best_checkpoint": str(best_ckpt_path),
                "epochs_requested": EPOCHS,
                "batch_size": BATCH_SIZE,
                "accum_steps": ACCUM_STEPS,
                "crop_size": CROP_SIZE,
                "scale_jitter": SCALE_JITTER,
                "hard_crop_candidates": HARD_CROP_CANDIDATES,
                "learning_rate": LR,
                "l1_weight": L1_WEIGHT,
                "ssim_weight": SSIM_WEIGHT,
                "fft_weight": FFT_WEIGHT,
                "amp": use_amp,
            }, indent=2), encoding="utf-8")

            if Path("/kaggle").exists():
                kaggle_working = Path("/kaggle/working")
                shutil.copy(best_ckpt_path, kaggle_working / best_ckpt_name)
                shutil.copy(summary_path, kaggle_working / summary_path.name)
                shutil.copy(preview_path, kaggle_working / preview_name)

        torch.save(
            checkpoint_payload(epoch, val_psnr, val_ssim, input_psnr, input_ssim),
            last_ckpt_path,
        )
        if Path("/kaggle").exists():
            shutil.copy(last_ckpt_path, Path("/kaggle/working") / last_ckpt_name)

    print(f"\nTraining complete. Best EMA PSNR: {best_psnr:.2f} dB")
    print(f"Best model: {best_ckpt_path}")

    summary_path.write_text(json.dumps({
        "dataset": args.dataset,
        "completed": True,
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "best_checkpoint": str(best_ckpt_path),
        "epochs_requested": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "crop_size": CROP_SIZE,
        "scale_jitter": SCALE_JITTER,
        "hard_crop_candidates": HARD_CROP_CANDIDATES,
        "learning_rate": LR,
        "l1_weight": L1_WEIGHT,
        "ssim_weight": SSIM_WEIGHT,
        "fft_weight": FFT_WEIGHT,
        "amp": use_amp,
    }, indent=2), encoding="utf-8")

    # ── Download link (Kaggle) ────────────────────────────────────────────────
    if Path("/kaggle").exists():
        kaggle_out = Path("/kaggle/working") / best_ckpt_name
        shutil.copy(best_ckpt_path, kaggle_out)
        shutil.copy(summary_path, Path("/kaggle/working") / summary_path.name)
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
