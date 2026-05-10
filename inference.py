"""
inference.py — Run the trained U-Net on a single moiré image.

Usage:
    # Basic — output only
    python inference.py --input path/to/moire_image.jpg

    # With ground truth for PSNR / SSIM evaluation
    python inference.py --input path/to/moire_image.jpg --gt path/to/clean_image.jpg

Output is saved to the results/ folder next to this script.
"""

import argparse
import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

# pytorch-msssim is only needed when --gt is provided
try:
    from pytorch_msssim import ssim as compute_ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False

from models.unet import UNet


# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
CKPT_PATH    = PROJECT_ROOT / "checkpoints" / "best_model.pth"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> UNet:
    """Load the U-Net and its weights from a checkpoint file."""
    model = UNet().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Support both raw state-dicts and our training checkpoint dicts
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    psnr  = checkpoint.get("val_psnr", None)
    info  = f"epoch {epoch}"
    if psnr is not None:
        info += f", val PSNR {psnr:.2f} dB"
    print(f"Loaded model from {ckpt_path.name} ({info})")
    return model


def load_image(path: Path) -> torch.Tensor:
    """Open an image and convert it to a [1, 3, H, W] float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img)   # [3, H, W], range [0, 1]
    return tensor.unsqueeze(0)   # add batch dimension


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] or [3, H, W] tensor in [0, 1] to a PIL image."""
    t = tensor.squeeze(0).clamp(0, 1)
    return TF.to_pil_image(t)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Tensors in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(input_path: Path, gt_path: Path | None, device: torch.device):
    model = load_model(CKPT_PATH, device)

    # Load and forward-pass the moiré image
    moire_tensor = load_image(input_path).to(device)
    pred_tensor  = model(moire_tensor).cpu()

    # Convert to PIL for display / saving
    moire_pil = tensor_to_pil(moire_tensor.cpu())
    pred_pil  = tensor_to_pil(pred_tensor)

    # Determine output filename
    stem       = input_path.stem
    out_path   = RESULTS_DIR / f"{stem}_cleaned.png"
    side_path  = RESULTS_DIR / f"{stem}_comparison.png"

    # Save the cleaned image
    pred_pil.save(out_path)
    print(f"Cleaned image saved → {out_path}")

    # ----- Metrics (only when ground truth is provided) --------------------
    if gt_path is not None:
        gt_tensor = load_image(gt_path).cpu()

        # Resize prediction to match GT if dimensions differ
        if pred_tensor.shape != gt_tensor.shape:
            pred_tensor = torch.nn.functional.interpolate(
                pred_tensor, size=gt_tensor.shape[2:], mode="bilinear", align_corners=False
            )

        psnr = compute_psnr(pred_tensor, gt_tensor)
        print(f"PSNR : {psnr:.2f} dB")

        if MSSSIM_AVAILABLE:
            ssim_val = compute_ssim(pred_tensor, gt_tensor, data_range=1.0, size_average=True).item()
            print(f"SSIM : {ssim_val:.4f}")
        else:
            print("SSIM : pytorch-msssim not installed — skipped")

    # ----- Side-by-side comparison figure ---------------------------------
    n_panels = 3 if gt_path else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    fig.suptitle(f"Moiré Removal — {input_path.name}", fontsize=14)

    axes[0].imshow(moire_pil)
    axes[0].set_title("Input (Moiré)")
    axes[0].axis("off")

    axes[1].imshow(pred_pil)
    axes[1].set_title("Output (Cleaned)")
    axes[1].axis("off")

    if gt_path:
        gt_pil = tensor_to_pil(gt_tensor)
        axes[2].imshow(gt_pil)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(side_path, dpi=150, bbox_inches="tight")
    print(f"Comparison figure saved → {side_path}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Moiré removal inference")
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the moiré input image",
    )
    parser.add_argument(
        "--gt", default=None, type=Path,
        help="(Optional) Path to the clean ground-truth image for metric evaluation",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=Path,
        help="(Optional) Path to a specific checkpoint (defaults to checkpoints/best_model.pth)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Allow overriding the default checkpoint path via CLI
    if args.checkpoint is not None:
        CKPT_PATH = args.checkpoint

    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CKPT_PATH}\n"
            "Train the model first with: python train.py"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_inference(
        input_path=args.input,
        gt_path=args.gt,
        device=device,
    )
