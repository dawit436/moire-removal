"""
prepare_fhdmi_kaggle.py — build a Kaggle-sized FHDMi subset.

This script validates that source/target filenames match by numeric index,
then copies paired images into a new train/test folder structure until the
requested size budget is reached.

Default output:
    D:/FHDMi_kaggle/data/{train,test}/{moire,clean}/

Default behavior:
    1. Reserve most of the budget for train by default.
    2. Add train pairs in sorted numeric order first, then test pairs.
  3. Renumber copied files sequentially from 0000 within each split.

Example:
    python prepare_fhdmi_kaggle.py --raw-root D:/FHDMi/data --budget-gb 19.5
    python prepare_fhdmi_kaggle.py --budget-gb 19.5
    python prepare_fhdmi_kaggle.py --out-root D:/FHDMi_kaggle/data
"""

import argparse
import shutil
import re
from pathlib import Path


RAW_ROOT = Path("D:/FHDMi_raw")
DEFAULT_OUT_ROOT = Path("D:/FHDMi_kaggle/data")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def extract_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot read numeric index from {path.name}")
    return int(match.group(1))


def list_images(directory: Path):
    return sorted((p for p in directory.iterdir() if p.is_file() and is_image(p)), key=lambda p: p.name)


def find_pair_dirs(split_dir: Path):
    candidates = [
        ("source", "target"),
        ("moire", "clean"),
        ("input", "gt"),
    ]
    for moire_name, clean_name in candidates:
        moire_dir = split_dir / moire_name
        clean_dir = split_dir / clean_name
        if moire_dir.exists() and clean_dir.exists():
            return moire_dir, clean_dir
    return None, None


def load_pairs(split_dir: Path, split_name: str):
    source_dir, target_dir = find_pair_dirs(split_dir)
    if source_dir is None or target_dir is None:
        raise FileNotFoundError(
            f"Missing paired directories under {split_dir}. "
            "Expected source/target or moire/clean."
        )

    source_files = list_images(source_dir)
    target_files = list_images(target_dir)
    if len(source_files) != len(target_files):
        raise RuntimeError(
            f"{split_name}: source={len(source_files)} target={len(target_files)}"
        )

    pairs = []
    mismatches = []
    for source_path, target_path in zip(source_files, target_files):
        source_index = extract_index(source_path)
        target_index = extract_index(target_path)
        if source_index != target_index:
            mismatches.append((source_path.name, target_path.name))
        pairs.append((source_path, target_path, source_index))

    if mismatches:
        preview = "\n".join(f"  {s} != {t}" for s, t in mismatches[:10])
        raise RuntimeError(f"{split_name}: filename indices do not match:\n{preview}")

    return pairs


def pair_size_bytes(source_path: Path, target_path: Path) -> int:
    return source_path.stat().st_size + target_path.stat().st_size


def copy_pairs(pairs, out_root: Path, split: str, budget_bytes: int, used_bytes: int):
    moire_out = out_root / split / "moire"
    clean_out = out_root / split / "clean"
    moire_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source_path, target_path, _ in pairs:
        pair_bytes = pair_size_bytes(source_path, target_path)
        if used_bytes + pair_bytes > budget_bytes:
            break

        out_name = f"{copied:04d}"
        shutil.copy2(source_path, moire_out / f"{out_name}_moire{source_path.suffix}")
        shutil.copy2(target_path, clean_out / f"{out_name}_gt{target_path.suffix}")
        used_bytes += pair_bytes
        copied += 1

    return copied, used_bytes


def copy_pairs_with_limit(pairs, out_root: Path, split: str, split_budget_bytes: int):
    return copy_pairs(pairs, out_root, split, split_budget_bytes, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_ROOT,
        help="Source FHDMi root containing train/test paired directories",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Destination root for the Kaggle-sized dataset",
    )
    parser.add_argument(
        "--budget-gb",
        type=float,
        default=19.5,
        help="Maximum total size for all copied files, in gigabytes",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Fraction of the budget reserved for train (remaining budget goes to test)",
    )
    args = parser.parse_args()

    if not args.raw_root.exists():
        raise SystemExit(f"[ERROR] {args.raw_root} does not exist")

    train_pairs = load_pairs(args.raw_root / "train", "train")
    test_pairs = load_pairs(args.raw_root / "test", "test")

    budget_bytes = int(args.budget_gb * 1e9)
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        (args.out_root / split / "moire").mkdir(parents=True, exist_ok=True)
        (args.out_root / split / "clean").mkdir(parents=True, exist_ok=True)

    used_bytes = 0
    print(f"RAW root   : {args.raw_root}")
    print(f"OUT root   : {args.out_root}")
    print(f"Budget     : {args.budget_gb:.2f} GB")
    print(f"Train ratio: {args.train_ratio:.2f}")
    print(f"Train pairs: {len(train_pairs):,}")
    print(f"Test pairs : {len(test_pairs):,}")

    train_budget_bytes = int(budget_bytes * args.train_ratio)
    test_budget_bytes = budget_bytes - train_budget_bytes

    train_copied, used_bytes = copy_pairs_with_limit(train_pairs, args.out_root, "train", train_budget_bytes)
    test_copied, test_used_bytes = copy_pairs_with_limit(test_pairs, args.out_root, "test", test_budget_bytes)
    used_bytes += test_used_bytes

    used_gb = used_bytes / 1e9
    remaining_gb = max(0.0, args.budget_gb - used_gb)

    print("\nDone")
    print(f"  Copied test  pairs : {test_copied:,}")
    print(f"  Copied train pairs : {train_copied:,}")
    print(f"  Total size         : {used_gb:.2f} GB")
    print(f"  Budget remaining   : {remaining_gb:.2f} GB")
    print(f"  Output             : {args.out_root}")

    if test_copied < len(test_pairs):
        print("  [WARN] Not all test pairs fit inside the budget")
    if train_copied < len(train_pairs):
        print("  [INFO] Train subset was truncated to stay under budget")


if __name__ == "__main__":
    main()
