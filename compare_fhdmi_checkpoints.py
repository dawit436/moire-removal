"""
Compare the two FHDMi MBCNN checkpoints on real moire images.

Usage:
    python compare_fhdmi_checkpoints.py --input test_images/4K_moire_test.jpg
    python compare_fhdmi_checkpoints.py --all
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from models.mbcnn import MBCNN


ROOT = Path(__file__).parent
CHECKPOINTS = [
    ("epoch003_psnr22.31", ROOT / "checkpoints" / "best_mbcnn_fhdmi.pth"),
    ("epoch005_psnr22.55", ROOT / "checkpoints" / "best_mbcnn_fhdmi (1).pth"),
]
RESULTS_DIR = ROOT / "results" / "fhdmi_checkpoint_compare"
PAD_MULTIPLE = 16


def checkpoint_state(ckpt):
    if isinstance(ckpt, dict):
        if "ema_state" in ckpt:
            return ckpt["ema_state"]
        if "model_state" in ckpt:
            return ckpt["model_state"]
    return ckpt


def load_model(path: Path, device: torch.device) -> MBCNN:
    ckpt = torch.load(path, map_location=device)
    model = MBCNN().to(device)
    model.load_state_dict(checkpoint_state(ckpt), strict=True)
    model.eval()

    if isinstance(ckpt, dict):
        print(
            f"{path.name}: epoch={ckpt.get('epoch')} "
            f"PSNR={ckpt.get('val_psnr', ckpt.get('best_psnr'))}"
        )
    return model


def resize_to_max(tensor: torch.Tensor, max_side: int) -> torch.Tensor:
    _, _, h, w = tensor.shape
    if max(h, w) <= max_side:
        return tensor
    scale = max_side / max(h, w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    return F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)


def pad_to_multiple(tensor: torch.Tensor, multiple: int):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, pad_h, pad_w


def unpad(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    h, w = tensor.shape[-2:]
    return tensor[:, :, : h - pad_h if pad_h else h, : w - pad_w if pad_w else w]


def label_image(image: Image.Image, label: str) -> Image.Image:
    labeled = image.copy()
    draw = ImageDraw.Draw(labeled)
    pad = 8
    box_h = 28
    draw.rectangle((0, 0, labeled.width, box_h), fill=(0, 0, 0))
    draw.text((pad, 7), label, fill=(255, 255, 255))
    return labeled


def make_grid(images: list[tuple[str, Image.Image]]) -> Image.Image:
    labeled = [label_image(img, label) for label, img in images]
    width = sum(img.width for img in labeled)
    height = max(img.height for img in labeled)
    grid = Image.new("RGB", (width, height), color=(245, 245, 245))
    x = 0
    for img in labeled:
        grid.paste(img, (x, 0))
        x += img.width
    return grid


@torch.no_grad()
def run_one(image_path: Path, models: list[tuple[str, MBCNN]], device: torch.device, max_side: int):
    image = Image.open(image_path).convert("RGB")
    original = TF.to_tensor(image).unsqueeze(0)
    resized = resize_to_max(original, max_side)
    padded, pad_h, pad_w = pad_to_multiple(resized.to(device), PAD_MULTIPLE)

    print(
        f"\n{image_path.name}: original={image.width}x{image.height}, "
        f"test={resized.shape[-1]}x{resized.shape[-2]}"
    )

    outputs = [("input", TF.to_pil_image(resized.squeeze(0).clamp(0, 1)))]
    for label, model in models:
        start = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(padded)
        elapsed = time.perf_counter() - start
        pred = unpad(pred.float().cpu(), pad_h, pad_w).clamp(0, 1)
        outputs.append((f"{label} ({elapsed:.1f}s)", TF.to_pil_image(pred.squeeze(0))))
        print(f"  {label}: {elapsed:.2f}s")

    grid = make_grid(outputs)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{image_path.stem}_fhdmi_compare_{max_side}.jpg"
    grid.save(out_path, quality=95)
    print(f"saved: {out_path}")


def image_files() -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(p for p in (ROOT / "test_images").iterdir() if p.suffix.lower() in exts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Run on every image in test_images.")
    parser.add_argument("--max-side", type=int, default=768)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models = []
    for label, path in CHECKPOINTS:
        if not path.exists():
            raise FileNotFoundError(path)
        models.append((label, load_model(path, device)))

    if args.all:
        inputs = image_files()
    elif args.input is not None:
        inputs = [args.input]
    else:
        inputs = [ROOT / "test_images" / "4K_moire_test.jpg"]

    for path in inputs:
        run_one(path, models, device, args.max_side)


if __name__ == "__main__":
    main()
