"""
evaluate.py — Full test-set evaluation of the trained U-Net moiré removal model.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model_unet.pth
    python evaluate.py --data-dir data/test --results-dir results
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in all environments
import matplotlib.pyplot as plt

from pytorch_msssim import ssim as _ssim_fn

from models.unet import UNet


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CKPT_PATH    = PROJECT_ROOT / "checkpoints" / "best_model_unet.pth"
DATA_DIR     = PROJECT_ROOT / "data" / "test"
RESULTS_DIR  = PROJECT_ROOT / "results"

DEVICE              = torch.device("cpu")
PAD_MULTIPLE        = 16       # model has 4 maxpool stages → 2^4 = 16
CROP_SIZE           = 512      # center-crop before inference for speed
FAILURE_THRESHOLD   = 18.0     # dB — images below this are flagged
TOP_N               = 5        # how many best/worst to print in summary
GRID_N              = 3        # best/worst count for the visual grid


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _extract_index(name: str):
    """Return leading integer from a filename stem, e.g. '0042_moire.jpg' → 42."""
    m = re.match(r"^(\d+)", Path(name).stem)
    return int(m.group(1)) if m else None


def find_pairs(moire_dir: Path, clean_dir: Path):
    """
    Match moire and clean images by their leading numeric index.
    Returns a list of (moire_path, clean_path) sorted by index.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def index_dir(d: Path):
        return {
            _extract_index(f.name): f
            for f in sorted(d.iterdir())
            if f.suffix.lower() in supported and _extract_index(f.name) is not None
        }

    moire_map = index_dir(moire_dir)
    clean_map = index_dir(clean_dir)
    common    = sorted(set(moire_map) & set(clean_map))

    if not common:
        raise RuntimeError(
            f"No paired images found.\n  moire: {moire_dir}\n  clean: {clean_dir}\n"
            "Filenames must share a leading numeric index (e.g. 0042_moire.jpg / 0042_gt.jpg)."
        )

    return [(moire_map[i], clean_map[i]) for i in common]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path) -> UNet:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train the model first:  python train.py"
        )

    model = UNet().to(DEVICE)

    # weights_only=False needed for checkpoints that include non-tensor scalars
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    state   = ckpt.get("model_state", ckpt)   # support both raw and wrapped dicts
    epoch   = ckpt.get("epoch", "?")
    val_psnr = ckpt.get("val_psnr", None)

    model.load_state_dict(state)
    model.eval()

    info = f"epoch {epoch}"
    if val_psnr is not None:
        info += f", val PSNR {val_psnr:.2f} dB"
    print(f"Loaded  : {ckpt_path.name} ({info})")
    return model


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def load_image(path: Path) -> torch.Tensor:
    """Load image → [1, 3, H, W] float32 tensor in [0, 1], center-cropped to CROP_SIZE."""
    img = Image.open(path).convert("RGB")
    img = TF.center_crop(img, CROP_SIZE)
    return TF.to_tensor(img).unsqueeze(0)


def pad_to_multiple(x: torch.Tensor, m: int):
    """Pad [1,C,H,W] so H and W are divisible by m. Returns (padded, (pad_h, pad_w))."""
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph == 0 and pw == 0:
        return x, (0, 0)
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (ph, pw)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] → H×W×3 uint8 numpy array."""
    return (t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Both tensors [1,3,H,W] in [0,1]."""
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM via pytorch_msssim. Both tensors [1,3,H,W] in [0,1]."""
    return _ssim_fn(pred, target, data_range=1.0, size_average=True).item()


# ---------------------------------------------------------------------------
# Single-image inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_one(model: UNet, moire_path: Path, clean_path: Path) -> dict:
    """
    Run the model on one image pair.
    Pads to a multiple of PAD_MULTIPLE, infers, then crops back to original size.
    Returns a result dict.
    """
    moire  = load_image(moire_path)   # [1, 3, H, W]
    clean  = load_image(clean_path)   # [1, 3, H, W]
    _, _, H, W = moire.shape

    padded, (ph, pw) = pad_to_multiple(moire, PAD_MULTIPLE)
    pred_padded = model(padded)
    pred = pred_padded[:, :, :H, :W]   # crop back to original size

    return {
        "filename" : moire_path.name,
        "psnr"     : compute_psnr(pred, clean),
        "ssim"     : compute_ssim(pred, clean),
        "moire_np" : tensor_to_uint8(moire),
        "pred_np"  : tensor_to_uint8(pred),
        "clean_np" : tensor_to_uint8(clean),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_comparison_grid(best: list, worst: list, out_path: Path) -> None:
    """
    6 rows × 3 columns: best GRID_N rows first, then worst GRID_N rows.
    Columns: [ Moiré Input | Model Output | Ground Truth ]
    """
    records = best + worst
    n_rows  = len(records)

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    if n_rows == 1:
        axes = axes[None, :]

    col_headers = ["Moiré Input", "Model Output", "Ground Truth"]
    for col, header in enumerate(col_headers):
        axes[0, col].set_title(header, fontsize=13, fontweight="bold", pad=6)

    for row, rec in enumerate(records):
        group = "BEST" if row < len(best) else "WORST"
        rank  = (row + 1) if row < len(best) else (row - len(best) + 1)
        label = f"{group} #{rank}  {rec['filename']}  —  PSNR: {rec['psnr']:.2f} dB"
        axes[row, 0].set_ylabel(label, fontsize=8, labelpad=6)

        for col, img in enumerate([rec["moire_np"], rec["pred_np"], rec["clean_np"]]):
            axes[row, col].imshow(img)
            axes[row, col].axis("off")

    fig.suptitle("Moiré Removal Evaluation — Best 3 vs Worst 3 by PSNR",
                 fontsize=15, y=1.005)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  : {out_path}")


def save_psnr_histogram(psnr_values: list, out_path: Path) -> None:
    """Histogram of PSNR values with mean and median lines."""
    arr    = np.asarray(psnr_values, dtype=np.float32)
    mean_v = float(np.mean(arr))
    med_v  = float(np.median(arr))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(arr, bins=30, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.axvline(mean_v, color="#dd4444", linewidth=2.0, linestyle="--",
               label=f"Mean:   {mean_v:.2f} dB")
    ax.axvline(med_v,  color="#22aa55", linewidth=2.0, linestyle=":",
               label=f"Median: {med_v:.2f} dB")
    ax.set_xlabel("PSNR (dB)", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("PSNR Distribution on Test Set", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  : {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(ckpt_path: Path, data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    moire_dir = data_dir / "moire"
    clean_dir = data_dir / "clean"
    pairs     = find_pairs(moire_dir, clean_dir)
    print(f"Dataset : {len(pairs)} paired test images  ({moire_dir})")

    model = load_model(ckpt_path)
    print(f"Device  : {DEVICE}\n")

    # ---- Per-image inference ------------------------------------------------
    results = []
    for moire_path, clean_path in tqdm(pairs, desc="Evaluating", unit="img"):
        rec = infer_one(model, moire_path, clean_path)
        results.append(rec)
        tqdm.write(
            f"  {rec['filename']:<32s}  "
            f"PSNR: {rec['psnr']:6.2f} dB   "
            f"SSIM: {rec['ssim']:.4f}"
        )

    # ---- Aggregate statistics -----------------------------------------------
    psnr_vals  = [r["psnr"] for r in results]
    ssim_vals  = [r["ssim"] for r in results]
    mean_psnr  = float(np.mean(psnr_vals))
    std_psnr   = float(np.std(psnr_vals))
    mean_ssim  = float(np.mean(ssim_vals))
    std_ssim   = float(np.std(ssim_vals))

    by_psnr = sorted(results, key=lambda r: r["psnr"], reverse=True)

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Mean PSNR : {mean_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"Mean SSIM : {mean_ssim:.4f} ± {std_ssim:.4f}")

    print(f"\nBest {TOP_N} images (highest PSNR):")
    for r in by_psnr[:TOP_N]:
        print(f"  {r['filename']:<32s}  {r['psnr']:.2f} dB")

    print(f"\nWorst {TOP_N} images (lowest PSNR):")
    for r in by_psnr[-TOP_N:][::-1]:   # worst first
        print(f"  {r['filename']:<32s}  {r['psnr']:.2f} dB")

    # ---- Failure analysis ---------------------------------------------------
    failures = [r for r in results if r["psnr"] < FAILURE_THRESHOLD]
    if failures:
        print(f"\nFailure analysis — images with PSNR < {FAILURE_THRESHOLD} dB "
              f"({len(failures)} image(s)):")
        for r in sorted(failures, key=lambda x: x["psnr"]):
            print(f"  {r['filename']:<32s}  {r['psnr']:.2f} dB")
    else:
        print(f"\nNo images fall below the {FAILURE_THRESHOLD} dB failure threshold.")

    # ---- Visual outputs -----------------------------------------------------
    best3  = by_psnr[:GRID_N]
    worst3 = by_psnr[-GRID_N:][::-1]   # worst first for display
    save_comparison_grid(best3, worst3, results_dir / "evaluation_grid.png")
    save_psnr_histogram(psnr_vals, results_dir / "psnr_distribution.png")

    # ---- Final report -------------------------------------------------------
    border = "=" * 40
    print(f"\n{border}")
    print("EVALUATION COMPLETE")
    print(f"Total images tested: {len(results)}")
    print(f"Mean PSNR:  {mean_psnr:.2f} dB")
    print(f"Mean SSIM:  {mean_ssim:.4f}")
    print(f"Best image:  {by_psnr[0]['filename']} → {by_psnr[0]['psnr']:.2f} dB")
    print(f"Worst image: {by_psnr[-1]['filename']} → {by_psnr[-1]['psnr']:.2f} dB")
    print(f"Images below {FAILURE_THRESHOLD} dB: {len(failures)}")
    print(f"Results saved to: {results_dir}")
    print(border)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate trained U-Net on the full test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   type=Path, default=CKPT_PATH,
                   help="Path to the model checkpoint.")
    p.add_argument("--data-dir",     type=Path, default=DATA_DIR,
                   help="Test data directory (must contain moire/ and clean/ subdirs).")
    p.add_argument("--results-dir",  type=Path, default=RESULTS_DIR,
                   help="Directory where output figures are written.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(args.checkpoint, args.data_dir, args.results_dir)
