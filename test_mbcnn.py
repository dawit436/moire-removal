"""
test_mbcnn.py — Evaluate the trained MBCNN model on TIP2018 test images and a
custom image.

Usage:
    python test_mbcnn.py
"""

import math
import random
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

try:
    from pytorch_msssim import ssim as compute_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False

from models.mbcnn import MBCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent
CKPT_PATH   = ROOT / "checkpoints" / "best_mbcnn_tip2018.pth"
MOIRE_DIR   = ROOT / "data" / "test" / "moire"
CLEAN_DIR   = ROOT / "data" / "test" / "clean"
CUSTOM_IMG  = ROOT / "test_images" / "new_moire.jpg"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_TEST       = 5
SEED         = 42
PAD_MULTIPLE = 16
MAX_SIDE     = 512   # longest-side cap before inference to stay within RAM

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> MBCNN:
    model = MBCNN().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)

    if "ema_state" in ckpt:
        state = ckpt["ema_state"]
        src   = "ema_state"
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
        src   = "model_state"
    else:
        state = ckpt
        src   = "raw state dict"

    model.load_state_dict(state)
    model.eval()

    epoch    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    val_psnr = ckpt.get("val_psnr", None) if isinstance(ckpt, dict) else None
    info     = f"epoch {epoch}"
    if val_psnr is not None:
        info += f", val PSNR {val_psnr:.2f} dB"
    print(f"Loaded MBCNN from {ckpt_path.name} ({src}, {info})")
    return model

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> torch.Tensor:
    """Returns [1, 3, H, W] float32 in [0, 1]."""
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def resize_to_max(t: torch.Tensor, max_side: int) -> torch.Tensor:
    """Downscale [1, C, H, W] so the longest side <= max_side; no-op if already small."""
    _, _, h, w = t.shape
    if max(h, w) <= max_side:
        return t
    scale = max_side / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    return torch.nn.functional.interpolate(
        t, size=(new_h, new_w), mode="bilinear", align_corners=False
    )


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(t.squeeze(0).clamp(0, 1))


def pad_to_multiple(t: torch.Tensor, multiple: int):
    """Pad [1, C, H, W] to nearest multiple; returns (padded_tensor, (pad_h, pad_w))."""
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, (pad_h, pad_w)


def crop_pad(t: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    h, w = t.shape[2], t.shape[3]
    return t[:, :, : h - pad_h if pad_h else h, : w - pad_w if pad_w else w]


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model: MBCNN, img_tensor: torch.Tensor, device: torch.device):
    """Resize to MAX_SIDE, pad, run inference, unpad; returns (output, resized_input, elapsed)."""
    resized = resize_to_max(img_tensor, MAX_SIDE)
    inp, (ph, pw) = pad_to_multiple(resized.to(device), PAD_MULTIPLE)
    t0      = time.perf_counter()
    out     = model(inp)
    elapsed = time.perf_counter() - t0
    out = crop_pad(out.cpu(), ph, pw)
    return out, resized, elapsed

# ---------------------------------------------------------------------------
# Test-set evaluation (with ground truth)
# ---------------------------------------------------------------------------

def run_test_set(model: MBCNN, device: torch.device):
    moire_paths = sorted(MOIRE_DIR.glob("*_moire.jpg"))
    random.seed(SEED)
    chosen = random.sample(moire_paths, min(N_TEST, len(moire_paths)))

    psnr_vals, ssim_vals, times = [], [], []
    rows = []  # (moire_pil, output_pil, gt_pil, psnr_val)

    for moire_path in chosen:
        idx      = moire_path.stem.split("_")[0]          # e.g. "0400"
        gt_path  = CLEAN_DIR / f"{idx}_gt.jpg"

        moire_t  = load_image(moire_path)
        gt_t     = load_image(gt_path)

        out_t, resized_moire, elapsed = infer(model, moire_t, device)
        times.append(elapsed)

        # GT resized to the same spatial size as the model output for fair metrics
        gt_resized = resize_to_max(gt_t, MAX_SIDE)

        p = psnr(out_t, gt_resized)
        psnr_vals.append(p)
        print(f"  {moire_path.name}  PSNR: {p:.2f} dB  ({elapsed*1000:.0f} ms)")

        if MSSSIM_AVAILABLE:
            s = compute_ssim(out_t, gt_resized, data_range=1.0, size_average=True).item()
            ssim_vals.append(s)

        rows.append((
            tensor_to_pil(resized_moire),
            tensor_to_pil(out_t),
            tensor_to_pil(gt_resized),
            p,
        ))

    # Build grid figure: N rows × 3 cols
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = [axes]

    for i, (moire_pil, out_pil, gt_pil, p) in enumerate(rows):
        axes[i][0].imshow(moire_pil);  axes[i][0].set_title("Moire Input");   axes[i][0].axis("off")
        axes[i][1].imshow(out_pil);    axes[i][1].set_title(f"MBCNN Output\nPSNR: {p:.2f} dB"); axes[i][1].axis("off")
        axes[i][2].imshow(gt_pil);     axes[i][2].set_title("Ground Truth");  axes[i][2].axis("off")

    plt.suptitle("MBCNN — TIP2018 Test Set", fontsize=15, y=1.01)
    plt.tight_layout()
    out_path = RESULTS_DIR / "mbcnn_test_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nTest grid saved → {out_path}")

    return psnr_vals, ssim_vals, times, fig


# ---------------------------------------------------------------------------
# Custom image (no ground truth)
# ---------------------------------------------------------------------------

def run_custom(model: MBCNN, device: torch.device):
    if not CUSTOM_IMG.exists():
        print(f"Custom image not found: {CUSTOM_IMG} — skipping.")
        return None

    moire_t                    = load_image(CUSTOM_IMG)
    out_t, resized_moire, elapsed = infer(model, moire_t, device)
    print(f"\nCustom image ({CUSTOM_IMG.name})  inference: {elapsed*1000:.0f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(tensor_to_pil(resized_moire)); axes[0].set_title("Moire Input");  axes[0].axis("off")
    axes[1].imshow(tensor_to_pil(out_t));         axes[1].set_title("MBCNN Output"); axes[1].axis("off")

    plt.suptitle(f"MBCNN — {CUSTOM_IMG.name}", fontsize=13)
    plt.tight_layout()
    out_path = RESULTS_DIR / "mbcnn_custom.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Custom result saved → {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = torch.device("cpu")
    print(f"Device: {device}\n")

    model = load_model(CKPT_PATH, device)

    print(f"\n--- Test set ({N_TEST} random images) ---")
    psnr_vals, ssim_vals, times, fig_test = run_test_set(model, device)

    fig_custom = run_custom(model, device)

    print("\n=== Summary ===")
    print(f"Mean PSNR  : {sum(psnr_vals)/len(psnr_vals):.2f} dB")
    if ssim_vals:
        print(f"Mean SSIM  : {sum(ssim_vals)/len(ssim_vals):.4f}")
    else:
        print("Mean SSIM  : N/A (install pytorch-msssim for SSIM)")
    print(f"Avg time   : {sum(times)/len(times)*1000:.0f} ms/image")

    plt.show()
