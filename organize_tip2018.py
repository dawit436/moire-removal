"""
organize_tip2018.py — Organize the raw TIP2018 download into training structure.

Usage:
    python organize_tip2018.py

Input:  D:/TIP2018_raw/   (downloaded + extracted from HuggingFace)
Output:
    D:/TIP2018/data/train/moire/   0000_moire.png ...
    D:/TIP2018/data/train/clean/   0000_gt.png ...
    D:/TIP2018/data/test/moire/
    D:/TIP2018/data/test/clean/

Key step: auto-crop the black cross-shaped frame from every image pair.
Split: 90 % train / 10 % test (seed=42) if no separate test split exists.
"""

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

RAW_ROOT = Path("D:/TIP2018_raw")
OUT_ROOT = Path("D:/TIP2018/data")
SEED     = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_image(p) -> bool:
    return Path(p).suffix in IMG_EXTS


def sorted_images(d: Path):
    # Use os.scandir so DirEntry.is_file() uses cached type — no per-file stat call
    result = []
    try:
        with os.scandir(d) as it:
            for entry in it:
                try:
                    if entry.is_file() and is_image(entry.name):
                        result.append(Path(entry.path))
                except OSError:
                    pass
    except OSError:
        pass
    return sorted(result)


def auto_crop(moire_img: Image.Image, clean_img: Image.Image):
    """
    Find the non-black bounding box in the clean image and crop both images
    to that region. TIP2018 images have a cross-shaped black border whose
    size varies per image, so we detect it per-image.
    """
    arr  = np.array(clean_img.convert("RGB"))
    mask = arr.max(axis=2) > 20          # any channel > 20 → non-black

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():  # entirely black → return unchanged
        return moire_img, clean_img

    r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1]) + 1
    c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1]) + 1
    box    = (c0, r0, c1, r1)
    return moire_img.crop(box), clean_img.crop(box)


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# ── Pair discovery ────────────────────────────────────────────────────────────

def find_pairs_in(base: Path):
    """
    Look for source/ + target/ directories under base (and one level of nesting).
    Returns [(moire_path, clean_path), ...] sorted by stem.
    """
    source_names = {"source", "moire", "input"}
    target_names = {"target", "clean", "gt"}

    search_dirs = [base]
    # handle trainData/trainData/ (flat nested)
    nested = base / base.name
    if nested.exists():
        search_dirs.insert(0, nested)
    # handle trainData/extracted/trainData/ (7-Zip extraction with subfolder)
    extracted_nested = base / "extracted" / base.name
    if extracted_nested.exists():
        search_dirs.insert(0, extracted_nested)
    # handle trainData/extracted/ directly
    extracted = base / "extracted"
    if extracted.exists():
        search_dirs.insert(0, extracted)

    for candidate in search_dirs:
        if not candidate.is_dir():
            continue
        children = {c.name.lower(): c for c in candidate.iterdir() if c.is_dir()}
        src_dir  = next((children[n] for n in source_names if n in children), None)
        tgt_dir  = next((children[n] for n in target_names if n in children), None)
        if src_dir and tgt_dir:
            src_files = sorted_images(src_dir)
            tgt_files = sorted_images(tgt_dir)

            # Match by stripped stem  (removes _source / _target suffixes)
            src_map = {f.stem.replace("_source", "").replace("_in", ""): f
                       for f in src_files}
            tgt_map = {f.stem.replace("_target", "").replace("_gt", ""): f
                       for f in tgt_files}
            common  = sorted(set(src_map) & set(tgt_map))
            if common:
                return [(src_map[k], tgt_map[k]) for k in common]

    return []


def discover_pairs():
    """
    Try standard TIP2018 layout (trainData/ + testData/) and fall back to
    scanning the whole RAW_ROOT.
    Returns (train_pairs, test_pairs).
    """
    train_pairs = find_pairs_in(RAW_ROOT / "trainData")
    test_pairs  = find_pairs_in(RAW_ROOT / "testData")

    if not train_pairs and not test_pairs:
        # Flat fallback: find any source+target pair in RAW_ROOT
        train_pairs = find_pairs_in(RAW_ROOT)

    return train_pairs, test_pairs


# ── Organize one split ────────────────────────────────────────────────────────

def organize_split(split: str, pairs: list, start_idx: int = 0):
    moire_out = OUT_ROOT / split / "moire"
    clean_out  = OUT_ROOT / split / "clean"
    moire_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)

    for i, (m_path, c_path) in enumerate(
        tqdm(pairs, desc=f"  {split}", unit="pair")
    ):
        idx = start_idx + i
        out_m = moire_out / f"{idx:04d}_moire.png"
        out_c = clean_out  / f"{idx:04d}_gt.png"
        if out_m.exists() and out_c.exists():
            continue  # already processed — resume-safe
        try:
            moire_img = Image.open(m_path).convert("RGB")
            clean_img  = Image.open(c_path).convert("RGB")
            moire_img, clean_img = auto_crop(moire_img, clean_img)
            moire_img.save(out_m)
            clean_img.save(out_c)
        except Exception as exc:
            print(f"  [WARN] Skipping {m_path.name}: {exc}")

    saved = len(list(moire_out.glob("*.png")))
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Organizing TIP2018 dataset")
    print("=" * 60)
    print(f"Source : {RAW_ROOT}")
    print(f"Output : {OUT_ROOT}")

    if not RAW_ROOT.exists():
        raise SystemExit(f"[ERROR] {RAW_ROOT} does not exist. Download + extract first.")



    train_pairs, test_pairs = discover_pairs()
    print(f"  Discovered {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")

    # If no pre-split test set, carve 10% from train (seed=42)
    if train_pairs and not test_pairs:
        print("  No separate test set — splitting 90/10 with seed=42")
        rng     = random.Random(SEED)
        indices = list(range(len(train_pairs)))
        rng.shuffle(indices)
        cut          = int(len(indices) * 0.9)
        test_pairs   = [train_pairs[i] for i in indices[cut:]]
        train_pairs  = [train_pairs[i] for i in indices[:cut]]
        print(f"  After split: {len(train_pairs)} train, {len(test_pairs)} test")

    if not train_pairs:
        raise SystemExit("[ERROR] No pairs found. Check that RAR files are extracted.")

    train_n = organize_split("train", train_pairs)
    test_n  = organize_split("test",  test_pairs)

    size = size_gb(OUT_ROOT)
    print(f"\n{'=' * 60}")
    print("TIP2018 organization complete")
    print(f"{'=' * 60}")
    print(f"  Train pairs : {train_n:,}")
    print(f"  Test  pairs : {test_n:,}")
    print(f"  Output size : {size:.1f} GB")
    print(f"  Location    : {OUT_ROOT}")


if __name__ == "__main__":
    main()
