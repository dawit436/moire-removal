"""
organize_all_datasets.py — Prepare FHDMi and TIP2018 for MBCNN training.

Usage:
    python organize_all_datasets.py            # processes both
    python organize_all_datasets.py --fhdmi
    python organize_all_datasets.py --tip2018
    python organize_all_datasets.py --both
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
FHDMI_RAW   = Path("D:/FHDMi_raw")
FHDMI_OUT   = Path("D:/FHDMi/data")

TIP_RAW     = Path("D:/TIP2018_raw")
TIP_OUT     = Path("D:/TIP2018/data")

SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def sorted_images(directory: Path):
    return sorted(f for f in directory.iterdir() if is_image(f))


def make_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def auto_crop(moire_img: Image.Image, clean_img: Image.Image):
    """
    Crop both images to the non-black bounding box of the clean image.
    Required for TIP2018 which has a cross-shaped black border.
    """
    arr = np.array(clean_img.convert("RGB"))
    mask = arr.sum(axis=2) > 10
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return moire_img, clean_img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    box = (int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1)
    return moire_img.crop(box), clean_img.crop(box)


# ── FHDMi ─────────────────────────────────────────────────────────────────────
def find_fhdmi_split(split: str):
    """
    Search FHDMI_RAW for a directory containing source/ and target/ (or moire/
    and clean/) subdirs for the given split. Returns (moire_dir, clean_dir) or
    (None, None) if not found.
    """
    # Walk up to 3 levels deep looking for source+target pairs
    for candidate in [FHDMI_RAW / split, FHDMI_RAW]:
        for sub in [candidate, *candidate.rglob("*")]:
            if not sub.is_dir():
                continue
            if (sub / "source").exists() and (sub / "target").exists():
                return sub / "source", sub / "target"
            if (sub / "moire").exists() and (sub / "clean").exists():
                return sub / "moire", sub / "clean"
    return None, None


def organize_fhdmi():
    print("\n" + "=" * 60)
    print("Organizing FHDMi ...")

    if not FHDMI_RAW.exists():
        print(f"  [ERROR] {FHDMI_RAW} does not exist. Download first.")
        return 0, 0, 0.0

    results = {}
    for split in ("train", "test"):
        moire_src_dir, clean_src_dir = find_fhdmi_split(split)
        if moire_src_dir is None:
            print(f"  [WARN] Cannot find {split} source/target under {FHDMI_RAW} — skipping")
            results[split] = 0
            continue

        moire_out = FHDMI_OUT / split / "moire"
        clean_out = FHDMI_OUT / split / "clean"
        make_dirs(moire_out, clean_out)

        moire_files = sorted_images(moire_src_dir)
        clean_files = sorted_images(clean_src_dir)

        n = min(len(moire_files), len(clean_files))
        if len(moire_files) != len(clean_files):
            print(f"  [WARN] {split}: moire={len(moire_files)}, clean={len(clean_files)} — using {n}")

        ext = moire_files[0].suffix if moire_files else ".jpg"

        for i in tqdm(range(n), desc=f"  FHDMi {split}"):
            shutil.copy2(moire_files[i], moire_out / f"{i:04d}_moire{ext}")
            shutil.copy2(clean_files[i], clean_out / f"{i:04d}_gt{ext}")

        print(f"  {split}: {n} pairs → {FHDMI_OUT / split}")
        results[split] = n

    size = size_gb(FHDMI_OUT) if FHDMI_OUT.exists() else 0.0
    return results.get("train", 0), results.get("test", 0), size


# ── TIP2018 ───────────────────────────────────────────────────────────────────
def find_tip_pairs(base_dir: Path):
    """Return [(moire_path, clean_path), ...] from a TIP2018 raw split directory."""
    # Handle both flat and nested (e.g. trainData/trainData/) structures
    search_dirs = [base_dir]
    nested = base_dir / base_dir.name
    if nested.exists():
        search_dirs.insert(0, nested)

    for candidate in search_dirs:
        src_dir = candidate / "source"
        tgt_dir = candidate / "target"
        if not (src_dir.exists() and tgt_dir.exists()):
            continue

        src_files = sorted_images(src_dir)
        tgt_files = sorted_images(tgt_dir)

        # Match by stem, stripping optional _source / _target suffixes
        src_map = {f.stem.replace("_source", ""): f for f in src_files}
        tgt_map = {f.stem.replace("_target", ""): f for f in tgt_files}
        common  = sorted(set(src_map) & set(tgt_map))
        if common:
            return [(src_map[k], tgt_map[k]) for k in common]

    return []


def organize_tip2018():
    print("\n" + "=" * 60)
    print("Organizing TIP2018 ...")

    if not TIP_RAW.exists():
        print(f"  [ERROR] {TIP_RAW} does not exist. Download first.")
        return 0, 0, 0.0

    train_pairs = find_tip_pairs(TIP_RAW / "trainData")
    test_pairs  = find_tip_pairs(TIP_RAW / "testData")

    print(f"  Found {len(train_pairs)} raw train pairs")
    print(f"  Found {len(test_pairs)} raw test pairs")

    # If no separate test set, carve 10 % from train with seed=42
    if not test_pairs and train_pairs:
        print("  No separate test set — splitting train 90/10 (seed=42)")
        rng = random.Random(SEED)
        indices = list(range(len(train_pairs)))
        rng.shuffle(indices)
        split_at    = int(len(indices) * 0.9)
        test_pairs  = [train_pairs[i] for i in indices[split_at:]]
        train_pairs = [train_pairs[i] for i in indices[:split_at]]

    for split, pairs in (("train", train_pairs), ("test", test_pairs)):
        if not pairs:
            print(f"  [WARN] No {split} pairs — skipping")
            continue

        moire_out = TIP_OUT / split / "moire"
        clean_out = TIP_OUT / split / "clean"
        make_dirs(moire_out, clean_out)

        for i, (m_path, c_path) in enumerate(tqdm(pairs, desc=f"  TIP2018 {split}")):
            moire_img = Image.open(m_path).convert("RGB")
            clean_img = Image.open(c_path).convert("RGB")
            moire_img, clean_img = auto_crop(moire_img, clean_img)
            moire_img.save(moire_out / f"{i:04d}_moire.png")
            clean_img.save(clean_out / f"{i:04d}_gt.png")

        print(f"  {split}: {len(pairs)} pairs → {TIP_OUT / split}")

    train_n = len(list((TIP_OUT / "train" / "moire").glob("*"))) if (TIP_OUT / "train" / "moire").exists() else 0
    test_n  = len(list((TIP_OUT / "test"  / "moire").glob("*"))) if (TIP_OUT / "test"  / "moire").exists() else 0
    size    = size_gb(TIP_OUT) if TIP_OUT.exists() else 0.0
    return train_n, test_n, size


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fhdmi",   action="store_true")
    parser.add_argument("--tip2018", action="store_true")
    parser.add_argument("--both",    action="store_true")
    args = parser.parse_args()

    run_all   = not (args.fhdmi or args.tip2018 or args.both)
    do_fhdmi  = args.fhdmi  or args.both or run_all
    do_tip    = args.tip2018 or args.both or run_all

    results = {}
    if do_fhdmi:
        results["fhdmi"] = organize_fhdmi()
    if do_tip:
        results["tip2018"] = organize_tip2018()

    print("\n\n================================")
    print("DATASET PREPARATION COMPLETE")
    print("================================")

    total_pairs = 0
    if "fhdmi" in results:
        tr, te, gb = results["fhdmi"]
        print(f"FHDMi:")
        print(f"  Train pairs: {tr:,} (expected: 9,981)")
        print(f"  Test  pairs: {te:,} (expected: 2,019)")
        print(f"  Total size : {gb:.1f} GB")
        total_pairs += tr + te

    if "tip2018" in results:
        tr, te, gb = results["tip2018"]
        print(f"TIP2018:")
        print(f"  Train pairs: {tr:,}")
        print(f"  Test  pairs: {te:,}")
        print(f"  Total size : {gb:.1f} GB")
        total_pairs += tr + te

    print(f"\nCombined total: {total_pairs:,} pairs")
    print("================================")


if __name__ == "__main__":
    main()
