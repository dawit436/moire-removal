"""
Self-contained Kaggle runner for training MBCNN on FHDMi.

Expected dataset layout under /kaggle/input/<dataset-slug>/:
    data/train/moire/*.jpg
    data/train/clean/*.jpg
    data/test/moire/*.jpg
    data/test/clean/*.jpg

The script also accepts the same layout without the data/ wrapper.
Edit the config block below for longer/shorter Kaggle runs.
"""

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


# ----------------------------- Config ---------------------------------------

SEED = int(os.environ.get("SEED", "42"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "1"))
CROP_SIZE = int(os.environ.get("CROP_SIZE", "512"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", "5"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "5"))
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.999"))
USE_AMP = os.environ.get("USE_AMP", "1") != "0"

L1_WEIGHT = 0.50
SSIM_WEIGHT = 0.20
FFT_WEIGHT = 0.30

KAGGLE_INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
CKPT_DIR = WORKING / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Dataset --------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def extract_index(path: Path):
    stem = path.stem
    digits = []
    for ch in stem:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    return int("".join(digits)) if digits else None


def index_files(directory: Path):
    mapping = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            idx = extract_index(path)
            if idx is not None:
                mapping[idx] = path
    return mapping


def has_pairs(root: Path):
    return (
        (root / "train" / "moire").is_dir()
        and (root / "train" / "clean").is_dir()
    )


def find_fhdmi_root():
    env_root = os.environ.get("FHDMI_DATA_ROOT")
    if env_root:
        root = Path(env_root)
        if has_pairs(root):
            return root
        raise FileNotFoundError(f"FHDMI_DATA_ROOT does not contain train/moire and train/clean: {root}")

    candidates = []
    for base in [KAGGLE_INPUT, *KAGGLE_INPUT.iterdir()]:
        candidates.extend([base, base / "data"])
    for train_dir in KAGGLE_INPUT.rglob("train"):
        candidates.append(train_dir.parent)

    seen = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if has_pairs(root):
            return root

    raise FileNotFoundError(
        "Could not find FHDMi pairs under /kaggle/input. "
        "Expected data/train/moire and data/train/clean."
    )


class MoireDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, crop_size: int = 512):
        self.root_dir = Path(root_dir)
        self.split = split
        self.crop_size = crop_size
        moire_map = index_files(self.root_dir / split / "moire")
        clean_map = index_files(self.root_dir / split / "clean")
        common = sorted(set(moire_map) & set(clean_map))
        if not common:
            raise RuntimeError(f"No paired images found for {split} under {self.root_dir}")
        self.pairs = [(moire_map[i], clean_map[i]) for i in common]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moire_path, clean_path = self.pairs[idx]
        moire_img = Image.open(moire_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        if self.split == "train":
            crop_size = random.choice([384, 512, 640])
            i, j, h, w = self._random_crop_params(moire_img, crop_size)
            moire_img = TF.crop(moire_img, i, j, h, w)
            clean_img = TF.crop(clean_img, i, j, h, w)
            if crop_size != self.crop_size:
                moire_img = TF.resize(moire_img, [self.crop_size, self.crop_size])
                clean_img = TF.resize(clean_img, [self.crop_size, self.crop_size])
            if torch.rand(1).item() > 0.5:
                moire_img = TF.hflip(moire_img)
                clean_img = TF.hflip(clean_img)
            if torch.rand(1).item() > 0.5:
                moire_img = TF.vflip(moire_img)
                clean_img = TF.vflip(clean_img)
            if torch.rand(1).item() > 0.5:
                angle = random.choice([90, 180, 270])
                moire_img = TF.rotate(moire_img, angle)
                clean_img = TF.rotate(clean_img, angle)
        else:
            moire_img = TF.center_crop(moire_img, self.crop_size)
            clean_img = TF.center_crop(clean_img, self.crop_size)

        return {
            "moire": TF.to_tensor(moire_img),
            "clean": TF.to_tensor(clean_img),
            "filename": moire_path.name,
        }

    def _random_crop_params(self, img, crop_size):
        w, h = img.size
        if w < crop_size or h < crop_size:
            raise ValueError(f"Image {w}x{h} is smaller than crop {crop_size}")
        top = torch.randint(0, h - crop_size + 1, (1,)).item()
        left = torch.randint(0, w - crop_size + 1, (1,)).item()
        return top, left, crop_size, crop_size


# ----------------------------- Model ----------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.GroupNorm(32, channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class FrequencyBranch(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int, n_blocks: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, branch_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(branch_ch) for _ in range(n_blocks)])

    def forward(self, x):
        return self.blocks(self.proj(x))


class LearnedDecomposition(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.blur = nn.Conv2d(in_ch, in_ch, 7, padding=3, groups=in_ch, bias=False)
        nn.init.constant_(self.blur.weight, 1.0 / 49.0)

    def forward(self, x):
        low = self.blur(x)
        high = x - low
        mid = 0.5 * (x + low)
        return low, mid, high


class ChannelAttention(nn.Module):
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

    def forward(self, x):
        weights = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class MBCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, branch_ch=128, n_blocks=5):
        super().__init__()
        self.decompose = LearnedDecomposition(in_channels)
        self.low_branch = FrequencyBranch(in_channels, branch_ch, n_blocks)
        self.mid_branch = FrequencyBranch(in_channels, branch_ch, n_blocks)
        self.high_branch = FrequencyBranch(in_channels, branch_ch, n_blocks)
        fused_ch = branch_ch * 3
        self.attention = ChannelAttention(fused_ch, reduction=8)
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        low, mid, high = self.decompose(x)
        fused = torch.cat(
            [self.low_branch(low), self.mid_branch(mid), self.high_branch(high)],
            dim=1,
        )
        residual = self.fusion(self.attention(fused))
        return torch.clamp(x + residual - 0.5, 0.0, 1.0)


# ----------------------------- Loss / metrics -------------------------------

def gaussian_window(window_size, sigma, channel, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window_2d = gauss[:, None] @ gauss[None, :]
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(pred, target, data_range=1.0, window_size=11):
    pred = pred.float()
    target = target.float()
    channel = pred.size(1)
    window = gaussian_window(window_size, 1.5, channel, pred.device, pred.dtype)
    padding = window_size // 2

    mu1 = F.conv2d(pred, window, padding=padding, groups=channel)
    mu2 = F.conv2d(target, window, padding=padding, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=padding, groups=channel) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return score.mean()


def fft_loss(pred, target):
    pred = pred.float()
    target = target.float()
    pred_mag = torch.log1p(torch.abs(torch.fft.fft2(pred)))
    target_mag = torch.log1p(torch.abs(torch.fft.fft2(target)))
    return F.l1_loss(pred_mag, target_mag)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_value = self.l1(pred, target)
        ssim_value = 1.0 - ssim(pred, target, data_range=1.0)
        fft_value = fft_loss(pred, target)
        return L1_WEIGHT * l1_value + SSIM_WEIGHT * ssim_value + FFT_WEIGHT * fft_value


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def batch_psnr(preds, targets):
    return float(np.mean([compute_psnr(preds[i], targets[i]) for i in range(preds.shape[0])]))


def batch_ssim(preds, targets):
    return float(ssim(preds, targets, data_range=1.0).item())


# ----------------------------- Training utils -------------------------------

class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.model = MBCNN().to(next(model.parameters()).device)
        self.model.load_state_dict(model.state_dict())
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.data.copy_(b.data)


def autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


def build_scaler(enabled):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def find_pretrained_checkpoint():
    env_path = os.environ.get("PRETRAINED_CKPT", "").strip()
    if env_path.lower() == "none":
        return None
    if env_path:
        return Path(env_path)

    candidates = sorted(KAGGLE_INPUT.rglob("*.pth"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print("Multiple .pth files found; set PRETRAINED_CKPT to choose one:")
        for path in candidates:
            print(f"  {path}")
    return None


def load_pretrained(model, device):
    ckpt_path = find_pretrained_checkpoint()
    if ckpt_path is None:
        print("No pretrained checkpoint selected.")
        return
    if not ckpt_path.exists():
        print(f"Pretrained checkpoint not found: {ckpt_path}")
        return

    print(f"Loading pretrained weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "ema_state" in ckpt:
        state = ckpt["ema_state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


def train_one_epoch(model, loader, optimizer, criterion, device, ema, scaler, use_amp):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), start=1):
        moire = batch["moire"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            pred = model(moire)
            loss = criterion(pred, clean)
            backward_loss = loss / ACCUM_STEPS

        if scaler is not None:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        if step % ACCUM_STEPS == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    for batch in tqdm(loader, desc="Val", leave=False):
        moire = batch["moire"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        pred = model(moire)
        total_loss += criterion(pred, clean).item()
        total_psnr += batch_psnr(pred.cpu(), clean.cpu())
        total_ssim += batch_ssim(pred.cpu(), clean.cpu())

    n = len(loader)
    return total_loss / n, total_psnr / n, total_ssim / n


# ----------------------------- Main -----------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    data_root = find_fhdmi_root()
    train_ds = MoireDataset(data_root, "train", crop_size=CROP_SIZE)
    val_split = "test" if (data_root / "test" / "moire").exists() else "train"
    val_ds = MoireDataset(data_root, val_split, crop_size=CROP_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    model = MBCNN().to(device)
    load_pretrained(model, device)
    ema = ModelEMA(model, EMA_DECAY)
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    warmup = LinearLR(optimizer, start_factor=1e-6 / LR, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    use_amp = USE_AMP and device.type == "cuda"
    scaler = build_scaler(use_amp)

    total_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "data_root": str(data_root),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "val_split": val_split,
        "params_m": round(total_params / 1e6, 3),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS,
        "crop_size": CROP_SIZE,
        "lr": LR,
        "amp": use_amp,
        "output_dir": str(CKPT_DIR),
    }, indent=2))

    best_psnr = -float("inf")
    best_path = CKPT_DIR / "best_mbcnn_fhdmi.pth"

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, ema, scaler, use_amp)
        val_loss, val_psnr, val_ssim = validate(ema.model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"psnr={val_psnr:.2f}dB | ssim={val_ssim:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | time={time.time() - started:.0f}s"
        )

        if epoch % SAVE_EVERY == 0:
            path = CKPT_DIR / f"mbcnn_fhdmi_epoch_{epoch:03d}.pth"
            torch.save({
                "epoch": epoch,
                "dataset": "fhdmi",
                "model_state": model.state_dict(),
                "ema_state": ema.model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, path)
            print(f"Saved checkpoint: {path}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "epoch": epoch,
                "dataset": "fhdmi",
                "model_state": model.state_dict(),
                "ema_state": ema.model.state_dict(),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, best_path)
            torch.save(ema.model.state_dict(), WORKING / "best_mbcnn_fhdmi_ema_state_dict.pth")
            print(f"New best PSNR: {best_psnr:.2f} dB -> {best_path}")

    summary_path = WORKING / "training_summary.json"
    summary_path.write_text(json.dumps({"best_psnr": best_psnr, "best_checkpoint": str(best_path)}, indent=2))
    print(f"Training complete. Best PSNR: {best_psnr:.2f} dB")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
