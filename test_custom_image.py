"""
test_custom_image.py — Run U-Net inference on a single custom moiré image.

No ground truth required — produces a 2-panel visual comparison only.

Usage:
    python test_custom_image.py

Outputs (in results/):
    custom_comparison_<timestamp>.png   — side-by-side figure
    custom_output_<timestamp>.png       — cleaned image alone
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from models.unet import UNet

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT  = Path(__file__).parent
CKPT_PATH     = PROJECT_ROOT / "checkpoints" / "best_model_unet.pth"
TEST_IMG_DIR  = PROJECT_ROOT / "test_images"
RESULTS_DIR   = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Helpers (adapted from inference.py)
# ---------------------------------------------------------------------------

def find_image(folder: Path) -> Path:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    candidates = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in supported]
    if not candidates:
        sys.exit(f"ERROR: No supported image found in {folder}")
    if len(candidates) > 1:
        print(f"WARNING: {len(candidates)} images found in {folder}; using the first: {candidates[0].name}")
    return candidates[0]


def load_model(ckpt_path: Path, device: torch.device) -> UNet:
    model = UNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()
    epoch    = ckpt.get("epoch", "?")
    val_psnr = ckpt.get("val_psnr", None)
    info     = f"epoch {epoch}" + (f", val PSNR {val_psnr:.2f} dB" if val_psnr else "")
    print(f"  Checkpoint  : {ckpt_path.name} ({info})")
    return model


def load_image(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Return ([1,3,H,W] float tensor in [0,1], (orig_h, orig_w))."""
    img    = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(img).unsqueeze(0)
    return tensor, (img.height, img.width)


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 16):
    """Pad H and W up to the nearest multiple. Returns (padded_tensor, (ph, pw))."""
    _, _, h, w = tensor.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return tensor, (0, 0)
    padded = torch.nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    return padded, (ph, pw)


def to_numpy_rgb(tensor: torch.Tensor) -> np.ndarray:
    """[1,3,H,W] or [3,H,W] → H×W×3 uint8."""
    return (tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 55)
    print("  U-Net Moiré Removal — Custom Image Test")
    print("=" * 55)

    # --- Validate inputs ---
    if not TEST_IMG_DIR.exists():
        sys.exit(f"ERROR: test_images/ folder not found at {TEST_IMG_DIR}")
    if not CKPT_PATH.exists():
        sys.exit(f"ERROR: Checkpoint not found at {CKPT_PATH}")

    img_path = find_image(TEST_IMG_DIR)
    print(f"  Input image : {img_path.name}")

    # --- Load image ---
    input_tensor, (orig_h, orig_w) = load_image(img_path)
    print(f"  Dimensions  : {orig_w} × {orig_h} px")

    # --- Resize if needed for CPU memory ---
    CPU_MAX_SIDE = 1024
    long_side = max(orig_h, orig_w)
    if long_side > CPU_MAX_SIDE:
        scale = CPU_MAX_SIDE / long_side
        inf_h = max(16, (int(orig_h * scale) // 16) * 16)
        inf_w = max(16, (int(orig_w * scale) // 16) * 16)
        print(f"  Resizing to : {inf_w} × {inf_h} px for CPU inference (long side capped at {CPU_MAX_SIDE})")
        inf_tensor = torch.nn.functional.interpolate(
            input_tensor, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
    else:
        inf_tensor = input_tensor
        inf_h, inf_w = orig_h, orig_w

    # --- Pad to multiple of 16 ---
    padded, (ph, pw) = pad_to_multiple(inf_tensor)

    # --- Load model ---
    device = torch.device("cpu")
    model  = load_model(CKPT_PATH, device)

    # --- Inference ---
    print("  Running inference ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            pred_padded = model(padded.to(device)).cpu()
        elapsed = time.perf_counter() - t0
        print(f"done.  ({elapsed:.1f} s)")
        success = True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"FAILED after {elapsed:.1f} s")
        print(f"  ERROR: {exc}")
        sys.exit(1)

    # Crop padding off
    pred_tensor = pred_padded[:, :, :inf_h, :inf_w]

    # --- Build output paths ---
    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = RESULTS_DIR / f"custom_comparison_{ts}.png"
    output_path     = RESULTS_DIR / f"custom_output_{ts}.png"

    # --- Save cleaned image ---
    cleaned_pil = Image.fromarray(to_numpy_rgb(pred_tensor))
    cleaned_pil.save(output_path)

    # --- Side-by-side figure (2 panels) ---
    input_np  = to_numpy_rgb(inf_tensor)
    output_np = to_numpy_rgb(pred_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Moiré Removal — {img_path.name}  ({inf_w}×{inf_h} px)",
        fontsize=14, fontweight="bold"
    )

    axes[0].imshow(input_np)
    axes[0].set_title("Input (with moiré)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(output_np)
    axes[1].set_title("Output (cleaned)", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # --- Console summary ---
    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Image tested    : {img_path.name}")
    print(f"  Dimensions used : {inf_w} × {inf_h} px")
    print(f"  Inference       : {'SUCCESS' if success else 'FAILED'}")
    print(f"  Inference time  : {elapsed:.2f} s")
    print(f"  Comparison      : {comparison_path}")
    print(f"  Cleaned output  : {output_path}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
