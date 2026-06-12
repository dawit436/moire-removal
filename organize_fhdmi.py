"""
organize_fhdmi.py — Extract and organize the raw FHDMi download.

FHDMi is downloaded as split tar.gz archives:
    D:\FHDMi_raw\train\source.tar.gz00 ... .gz12
    D:\FHDMi_raw\train\target.tar.gz00 ... .gz10
    D:\FHDMi_raw\test\source.tar.gz00  ... .gz02
    D:\FHDMi_raw\test\target.tar.gz00  ... .gz02

This script:
  1. Combines the split parts and extracts them with Python tarfile
  2. Renames images sequentially (0000_moire.jpg, 0000_gt.jpg, ...)
  3. Saves to D:\FHDMi\data\{train,test}\{moire,clean}\

Usage:
    python organize_fhdmi.py [--skip-extract]   # --skip-extract if already done
"""

import argparse
import io
import shutil
import tarfile
from pathlib import Path

from tqdm import tqdm

RAW_ROOT  = Path("D:/FHDMi_raw")
EXTR_ROOT = Path("D:/FHDMi_extracted")   # intermediate extracted images
OUT_ROOT  = Path("D:/FHDMi/data")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def sorted_images(d: Path):
    return sorted(f for f in d.iterdir() if f.is_file() and is_image(f))


def size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# ── Split-archive extraction ──────────────────────────────────────────────────

def find_split_groups(directory: Path):
    """
    Find groups of split-archive files.
    Returns dict like {"source": [p00, p01, ...], "target": [...]}
    Handles:
      - .tar.gz00 / .gz01 … style (FHDMi)
      - .part1.rar / .part2.rar … style
    """
    files = sorted(directory.iterdir()) if directory.exists() else []
    groups = {}
    for f in files:
        if not f.is_file():
            continue
        stem = f.name
        # Strip the numeric suffix (.gz00, .gz01 …)
        for sep in (".gz", ".part"):
            idx = stem.rfind(sep)
            if idx != -1:
                tail = stem[idx + len(sep):]
                if tail.isdigit():
                    base = stem[:idx]
                    groups.setdefault(base, []).append(f)
                    break
    return {k: sorted(v) for k, v in groups.items()}


def combine_and_extract(parts: list, out_dir: Path, desc: str):
    """
    Concatenate split gzip parts in memory and extract as tar.gz.
    Works for files that fit in a streaming model (does not load all into RAM).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    class MultiReader(io.RawIOBase):
        """Streams multiple files as one continuous byte stream."""
        def __init__(self, paths):
            self._paths = list(paths)
            self._idx   = 0
            self._fh    = open(self._paths[0], "rb") if self._paths else None

        def readinto(self, buf):
            while self._fh is not None:
                n = self._fh.readinto(buf)
                if n:
                    return n
                self._fh.close()
                self._idx += 1
                if self._idx < len(self._paths):
                    self._fh = open(self._paths[self._idx], "rb")
                else:
                    self._fh = None
            return 0

        def readable(self):
            return True

    total_bytes = sum(p.stat().st_size for p in parts)
    print(f"  Extracting {desc}  ({total_bytes / 1e9:.2f} GB combined) → {out_dir}")

    stream = io.BufferedReader(MultiReader(parts), buffer_size=1 << 23)  # 8 MB buf
    with tarfile.open(fileobj=stream, mode="r:gz") as tf:
        members = tf.getmembers()
        for m in tqdm(members, desc=f"    {desc}", unit="file", leave=False):
            tf.extract(m, out_dir, set_attrs=False)

    print(f"    Done — {len(members)} files extracted")


# ── Extraction driver ─────────────────────────────────────────────────────────

def extract_all():
    """Extract all split tar.gz archives from RAW_ROOT into EXTR_ROOT."""
    print("=" * 60)
    print("Phase 1: Extracting split archives")
    print("=" * 60)

    for split in ("train", "test"):
        split_dir = RAW_ROOT / split
        if not split_dir.exists():
            print(f"  [WARN] {split_dir} not found — skipping")
            continue

        groups = find_split_groups(split_dir)
        if not groups:
            print(f"  [INFO] No split archives in {split_dir} (may already be images)")
            continue

        for base_name, parts in groups.items():
            out_dir = EXTR_ROOT / split / base_name
            combine_and_extract(parts, out_dir, f"{split}/{base_name}")


# ── Image discovery ───────────────────────────────────────────────────────────

def find_images_in(base: Path):
    """Recursively find all images under base, sorted."""
    return sorted(p for p in base.rglob("*") if p.is_file() and is_image(p))


def find_split_pairs(split: str):
    """
    Returns (moire_files, clean_files) for a given split.
    Searches both EXTR_ROOT (after extraction) and RAW_ROOT (direct images).
    """
    source_names = {"source", "moire", "input", "degraded"}
    target_names = {"target", "clean", "gt", "ground_truth"}

    for search_root in [EXTR_ROOT / split, RAW_ROOT / split]:
        if not search_root.exists():
            continue
        children = {c.name.lower(): c for c in search_root.iterdir() if c.is_dir()}
        moire_dir = next((children[n] for n in source_names if n in children), None)
        clean_dir  = next((children[n] for n in target_names if n in children), None)
        if moire_dir and clean_dir:
            m = find_images_in(moire_dir)
            c = find_images_in(clean_dir)
            if m and c:
                return m, c

    return [], []


# ── Organize ──────────────────────────────────────────────────────────────────

def organize_split(split: str):
    moire_files, clean_files = find_split_pairs(split)

    if not moire_files or not clean_files:
        print(f"  [ERROR] No images found for {split} split.")
        print(f"          Check extraction completed successfully.")
        return 0

    n = min(len(moire_files), len(clean_files))
    if len(moire_files) != len(clean_files):
        print(f"  [WARN] {split}: moire={len(moire_files)}, clean={len(clean_files)} → using {n}")

    moire_out = OUT_ROOT / split / "moire"
    clean_out  = OUT_ROOT / split / "clean"
    moire_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)

    ext = moire_files[0].suffix

    for i in tqdm(range(n), desc=f"  Organizing {split}", unit="pair"):
        shutil.copy2(moire_files[i], moire_out / f"{i:04d}_moire{ext}")
        shutil.copy2(clean_files[i],  clean_out  / f"{i:04d}_gt{ext}")

    return n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip extraction (use if already extracted to D:\\FHDMi_extracted\\)"
    )
    args = parser.parse_args()

    if not RAW_ROOT.exists():
        raise SystemExit(f"[ERROR] {RAW_ROOT} does not exist — run gdown first.")

    if not args.skip_extract:
        extract_all()
    else:
        print("Skipping extraction (--skip-extract)")

    print("\n" + "=" * 60)
    print("Phase 2: Organizing into training structure")
    print("=" * 60)
    print(f"Output: {OUT_ROOT}")

    train_n = organize_split("train")
    test_n  = organize_split("test")

    size = size_gb(OUT_ROOT)

    print(f"\n{'=' * 60}")
    print("FHDMi organization complete")
    print(f"{'=' * 60}")
    print(f"  Train pairs : {train_n:,}  (expected 9,981)")
    print(f"  Test  pairs : {test_n:,}  (expected 2,019)")
    print(f"  Output size : {size:.1f} GB")
    print(f"  Location    : {OUT_ROOT}")


if __name__ == "__main__":
    main()
