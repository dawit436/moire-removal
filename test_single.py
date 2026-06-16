"""
test_single.py — Single-image MBCNN inference with side-by-side visual comparison.
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

from models.mbcnn import MBCNN

ROOT        = Path(__file__).parent
CKPT_PATH   = ROOT / "checkpoints" / "best_mbcnn_tip2018.pth"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_SIDE     = 512
PAD_MULTIPLE = 16


def load_model(ckpt_path: Path, device: torch.device) -> MBCNN:
    model = MBCNN().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)

    if "ema_state" in ckpt:
        state = ckpt["ema_state"]
    elif "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()

    epoch    = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    val_psnr = ckpt.get("val_psnr", None) if isinstance(ckpt, dict) else None
    info     = f"epoch {epoch}" + (f", val PSNR {val_psnr:.2f} dB" if val_psnr else "")
    print(f"Loaded MBCNN from {ckpt_path.name} ({info})")
    return model


def load_image(path: Path) -> torch.Tensor:
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


if __name__ == "__main__":
    # Find the single image in test_images/
    test_images = list((ROOT / "test_images").glob("*"))
    test_images = [p for p in test_images if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")]
    if not test_images:
        raise FileNotFoundError("No images found in test_images/")
    img_path = test_images[0]
    print(f"Input : {img_path.name}")

    device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_model(CKPT_PATH, device)

    # Load and preprocess
    img_t   = load_image(img_path)
    resized = resize_to_max(img_t, MAX_SIDE)
    padded, (ph, pw) = pad_to_multiple(resized.to(device), PAD_MULTIPLE)

    print(f"Input size : {img_t.shape[3]}x{img_t.shape[2]}  ->  resized {resized.shape[3]}x{resized.shape[2]}  ->  padded {padded.shape[3]}x{padded.shape[2]}")

    # Inference
    with torch.no_grad():
        t0  = time.perf_counter()
        out = model(padded)
        elapsed = time.perf_counter() - t0

    out = crop_pad(out.cpu(), ph, pw)
    print(f"Inference time: {elapsed * 1000:.1f} ms")

    moire_pil = tensor_to_pil(resized)
    output_pil = tensor_to_pil(out)

    # Side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"MBCNN — {img_path.name}  ({elapsed*1000:.0f} ms)", fontsize=13)

    axes[0].imshow(moire_pil)
    axes[0].set_title("Moire Input", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(output_pil)
    axes[1].set_title("Cleaned Output", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    out_path = RESULTS_DIR / "single_test.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.show()
