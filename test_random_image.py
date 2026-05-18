"""
test_random_image.py — Pick a random test image, run U-Net inference, save comparison.

Outputs (in results/):
    comparison_<timestamp>_idx<NNNN>.png   — 3-panel side-by-side figure
    output_<timestamp>_idx<NNNN>.png       — cleaned image alone
"""

import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

try:
    from pytorch_msssim import ssim as _msssim
    def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
        return _msssim(pred, gt, data_range=1.0, size_average=True).item()
    SSIM_BACKEND = "pytorch_msssim"
except ImportError:
    try:
        from skimage.metrics import structural_similarity as _sk_ssim
        def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
            p = pred.squeeze(0).permute(1, 2, 0).numpy()
            g = gt.squeeze(0).permute(1, 2, 0).numpy()
            return _sk_ssim(p, g, data_range=1.0, channel_axis=2)
        SSIM_BACKEND = "skimage"
    except ImportError:
        compute_ssim = None
        SSIM_BACKEND = None

from models.unet import UNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# For CPU inference, cap the long edge to avoid OOM on high-res images.
# Must be a multiple of 16 (U-Net has 4 downsampling stages).
CPU_MAX_SIDE = 1024

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CKPT_PATH    = PROJECT_ROOT / "checkpoints" / "best_model_unet.pth"
TEST_MOIRE   = PROJECT_ROOT / "data" / "test" / "moire"
TEST_CLEAN   = PROJECT_ROOT / "data" / "test" / "clean"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_index(name: str) -> int | None:
    m = re.match(r"^(\d+)", Path(name).stem)
    return int(m.group(1)) if m else None


def _pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    """Pad H and W up to the nearest multiple of `multiple`. Returns (padded, (ph, pw))."""
    _, _, h, w = tensor.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return tensor, (0, 0)
    # F.pad takes (left, right, top, bottom)
    padded = torch.nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    return padded, (ph, pw)


def load_model(device: torch.device) -> UNet:
    model = UNet().to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_psnr = ckpt.get("val_psnr", None)
    info = f"epoch {epoch}" + (f", val PSNR {val_psnr:.2f} dB" if val_psnr else "")
    print(f"Loaded checkpoint: {CKPT_PATH.name} ({info})")
    return model


def load_image(path: Path) -> torch.Tensor:
    """Return [1, 3, H, W] float tensor in [0, 1]."""
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] → HxWx3 uint8 numpy array."""
    return (t.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Find paired test images ---
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    moire_map = {
        _extract_index(f.name): f
        for f in sorted(TEST_MOIRE.iterdir())
        if f.suffix.lower() in supported and _extract_index(f.name) is not None
    }
    clean_map = {
        _extract_index(f.name): f
        for f in sorted(TEST_CLEAN.iterdir())
        if f.suffix.lower() in supported and _extract_index(f.name) is not None
    }
    common_indices = sorted(set(moire_map) & set(clean_map))
    if not common_indices:
        sys.exit("ERROR: No paired test images found. Check data/test/moire and data/test/clean.")

    chosen_idx = random.choice(common_indices)
    moire_path = moire_map[chosen_idx]
    clean_path = clean_map[chosen_idx]
    print(f"Selected image pair: {moire_path.name}  +  {clean_path.name}")

    # --- Load model ---
    device = torch.device("cpu")
    model = load_model(device)

    # --- Load images ---
    moire_tensor = load_image(moire_path)   # [1,3,H,W]
    gt_tensor    = load_image(clean_path)   # [1,3,H,W]
    orig_h, orig_w = moire_tensor.shape[2], moire_tensor.shape[3]
    print(f"Image size: {orig_w} × {orig_h}")

    # --- Resize for inference if image exceeds CPU_MAX_SIDE ---
    long_side = max(orig_h, orig_w)
    if long_side > CPU_MAX_SIDE:
        scale = CPU_MAX_SIDE / long_side
        inf_h = int(orig_h * scale)
        inf_w = int(orig_w * scale)
        # Snap to nearest multiple of 16
        inf_h = max(16, (inf_h // 16) * 16)
        inf_w = max(16, (inf_w // 16) * 16)
        print(f"Resizing to {inf_w} × {inf_h} for CPU inference (max side {CPU_MAX_SIDE})")
        moire_inf = torch.nn.functional.interpolate(
            moire_tensor, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
        gt_inf = torch.nn.functional.interpolate(
            gt_tensor, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
        resized = True
    else:
        moire_inf, gt_inf = moire_tensor, gt_tensor
        resized = False

    # --- Pad to multiple of 16 if needed (handles non-max-side images) ---
    padded, (ph, pw) = _pad_to_multiple(moire_inf)
    inf_h2, inf_w2 = moire_inf.shape[2], moire_inf.shape[3]

    # --- Inference ---
    print("Running inference...", end=" ", flush=True)
    with torch.no_grad():
        pred_padded = model(padded.to(device)).cpu()
    print("done.")

    # Crop padding back off
    pred_tensor = pred_padded[:, :, :inf_h2, :inf_w2]

    # Use the (possibly resized) GT for metrics
    gt_tensor = gt_inf

    # --- Metrics ---
    psnr = compute_psnr(pred_tensor, gt_tensor)
    print(f"PSNR : {psnr:.2f} dB")

    ssim_str = "N/A"
    if compute_ssim is not None:
        ssim_val = compute_ssim(pred_tensor, gt_tensor)
        ssim_str = f"{ssim_val:.4f}"
        print(f"SSIM : {ssim_str}  (backend: {SSIM_BACKEND})")
    else:
        print("SSIM : skipped (install pytorch_msssim or scikit-image)")

    # --- Build output filenames ---
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx4 = f"{chosen_idx:04d}"
    comparison_path = RESULTS_DIR / f"comparison_{ts}_idx{idx4}.png"
    output_path     = RESULTS_DIR / f"output_{ts}_idx{idx4}.png"

    # --- Save cleaned image alone ---
    cleaned_pil = Image.fromarray(tensor_to_numpy(pred_tensor))
    cleaned_pil.save(output_path)

    # --- Side-by-side comparison figure ---
    moire_np = tensor_to_numpy(moire_tensor)
    pred_np  = tensor_to_numpy(pred_tensor)
    gt_np    = tensor_to_numpy(gt_tensor)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    overall_title = (
        f"Moiré Removal — {moire_path.name}\n"
        f"PSNR: {psnr:.2f} dB    SSIM: {ssim_str}"
    )
    fig.suptitle(overall_title, fontsize=14, fontweight="bold")

    panels = [
        (axes[0], moire_np, "Moiré Input"),
        (axes[1], pred_np,  "Model Output (Cleaned)"),
        (axes[2], gt_np,    "Ground Truth"),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Console summary ---
    print(f"\nResults saved:")
    print(f"  Comparison : {comparison_path}")
    print(f"  Output     : {output_path}")


if __name__ == "__main__":
    main()
