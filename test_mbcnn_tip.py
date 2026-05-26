"""
test_mbcnn_tip.py — Evaluate MBCNN on the full TIP2018 test set (external drive).

Usage:
    python test_mbcnn_tip.py
"""

import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
# Config
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent
CKPT_PATH   = ROOT / "checkpoints" / "best_mbcnn_tip2018.pth"
SOURCE_DIR  = Path(r"D:\tip2018\testData\testData\source")
TARGET_DIR  = Path(r"D:\tip2018\testData\testData\target")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLE     = 10
SEED         = 42
MAX_SIDE     = 512
PAD_MULTIPLE = 16

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> MBCNN:
    model = MBCNN().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)

    if "ema_state" in ckpt:
        state, src = ckpt["ema_state"], "ema_state"
    elif "model_state" in ckpt:
        state, src = ckpt["model_state"], "model_state"
    else:
        state, src = ckpt, "raw state dict"

    model.load_state_dict(state)
    model.eval()

    epoch    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    val_psnr = ckpt.get("val_psnr", None) if isinstance(ckpt, dict) else None
    info     = f"epoch {epoch}" + (f", val PSNR {val_psnr:.2f} dB" if val_psnr else "")
    print(f"Loaded MBCNN from {ckpt_path.name} ({src}, {info})\n")
    return model

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: Path) -> torch.Tensor:
    """Returns [1, 3, H, W] float32 in [0, 1]."""
    return TF.to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


def resize_to_max(t: torch.Tensor, max_side: int) -> torch.Tensor:
    _, _, h, w = t.shape
    if max(h, w) <= max_side:
        return t
    scale = max_side / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    return F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)


def pad_to_multiple(t: torch.Tensor, multiple: int):
    _, _, h, w = t.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, (pad_h, pad_w)


def crop_pad(t: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    h, w = t.shape[2], t.shape[3]
    return t[:, :, : h - pad_h if pad_h else h, : w - pad_w if pad_w else w]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    return TF.to_pil_image(t.squeeze(0).clamp(0, 1))


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model: MBCNN, img_t: torch.Tensor, device: torch.device):
    """Resize → pad → forward → unpad. Returns (output, resized_input, elapsed_s)."""
    resized        = resize_to_max(img_t, MAX_SIDE)
    padded, (ph, pw) = pad_to_multiple(resized.to(device), PAD_MULTIPLE)
    t0             = time.perf_counter()
    out            = model(padded)
    elapsed        = time.perf_counter() - t0
    out            = crop_pad(out.cpu(), ph, pw)
    return out, resized, elapsed

# ---------------------------------------------------------------------------
# Pair discovery
# ---------------------------------------------------------------------------

def find_pairs():
    """
    Match source/target by replacing '_source' with '_target' in the stem.
    Returns list of (source_path, target_path).
    """
    sources = sorted(SOURCE_DIR.glob("*_source.png"))
    pairs   = []
    for src in sources:
        tgt = TARGET_DIR / src.name.replace("_source.png", "_target.png")
        if tgt.exists():
            pairs.append((src, tgt))
    return pairs

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for d, label in [(SOURCE_DIR, "source"), (TARGET_DIR, "target")]:
        if not d.exists():
            raise FileNotFoundError(f"TIP2018 {label} directory not found: {d}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model(CKPT_PATH, device)

    all_pairs = find_pairs()
    if not all_pairs:
        raise RuntimeError(f"No matched source/target pairs found in {SOURCE_DIR}")
    print(f"Found {len(all_pairs)} pairs — sampling {N_SAMPLE} (seed={SEED})\n")

    random.seed(SEED)
    sample = random.sample(all_pairs, min(N_SAMPLE, len(all_pairs)))

    results = []  # (psnr, ssim_or_None, moire_pil, out_pil, gt_pil, name)

    for src_path, tgt_path in sample:
        src_t = load_image(src_path)
        tgt_t = load_image(tgt_path)

        out_t, resized_src, elapsed = infer(model, src_t, device)
        gt_resized                  = resize_to_max(tgt_t, MAX_SIDE)

        p = compute_psnr(out_t, gt_resized)
        s = None
        if MSSSIM_AVAILABLE:
            s = compute_ssim(out_t, gt_resized, data_range=1.0, size_average=True).item()

        ssim_str = f"  SSIM: {s:.4f}" if s is not None else ""
        print(f"  {src_path.name}  PSNR: {p:.2f} dB{ssim_str}  ({elapsed*1000:.0f} ms)")

        results.append((
            p, s,
            tensor_to_pil(resized_src),
            tensor_to_pil(out_t),
            tensor_to_pil(gt_resized),
            src_path.stem,
        ))

    # ---- Sort by PSNR, pick best 3 and worst 3 ----------------------------
    results.sort(key=lambda x: x[0])
    worst3 = results[:3]
    best3  = results[-3:][::-1]   # best first
    display_rows = best3 + worst3
    labels = ["BEST"] * 3 + ["WORST"] * 3

    # ---- Grid: 6 rows × 3 cols --------------------------------------------
    n = len(display_rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))

    for i, ((p, s, moire_pil, out_pil, gt_pil, name), tag) in enumerate(
        zip(display_rows, labels)
    ):
        ssim_str = f"  SSIM {s:.4f}" if s is not None else ""
        row_label = f"[{tag}] {name}"

        axes[i][0].imshow(moire_pil)
        axes[i][0].set_title(f"Moire Input\n{row_label}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(out_pil)
        axes[i][1].set_title(f"MBCNN Output\nPSNR {p:.2f} dB{ssim_str}", fontsize=8)
        axes[i][1].axis("off")

        axes[i][2].imshow(gt_pil)
        axes[i][2].set_title("Ground Truth", fontsize=8)
        axes[i][2].axis("off")

    plt.suptitle("MBCNN — TIP2018 Test Set (best 3 / worst 3)", fontsize=14, y=1.005)
    plt.tight_layout()

    out_path = RESULTS_DIR / "mbcnn_tip2018_test.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nGrid saved → {out_path}")

    # ---- Summary ----------------------------------------------------------
    psnr_vals = [r[0] for r in results]
    ssim_vals = [r[1] for r in results if r[1] is not None]

    print("\n=== Summary ===")
    print(f"Samples      : {len(results)}")
    print(f"Mean PSNR    : {sum(psnr_vals)/len(psnr_vals):.2f} dB")
    print(f"Best  PSNR   : {max(psnr_vals):.2f} dB")
    print(f"Worst PSNR   : {min(psnr_vals):.2f} dB")
    if ssim_vals:
        print(f"Mean SSIM    : {sum(ssim_vals)/len(ssim_vals):.4f}")
    else:
        print("Mean SSIM    : N/A (install pytorch-msssim)")

    plt.show()
