"""
dataset_analysis_full.py — Comprehensive quality audit for D:\\dataset\\train\\
Optimised: PIL draft() JPEG fast-decode + multiprocessing SSIM pass.
Run:  python dataset_analysis_full.py
"""

import sys
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
from scipy import fft as scipy_fft
from skimage.metrics import structural_similarity as ssim_metric
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
DATASET_ROOT   = Path("D:/dataset/train")
RESULTS_DIR    = Path(__file__).parent / "results"
SSIM_W, SSIM_H = 512, 384      # draft() decode target (nearest DCT scale used)
SSIM_THRESH_EX = 0.75
SSIM_THRESH_GD = 0.60
SSIM_THRESH_WN = 0.40
SHIFT_THRESH   = 20            # pixels
GT_SAMPLE_N    = 50
PEAK_RATIO_BAD = 3.0
BRIGHTNESS_GAP = 30
N_WORKERS      = max(1, min(cpu_count() - 1, 6))
# ─────────────────────────────────────────────────────────────────────────────

# ── helpers (must be at module level so pickling works on Windows) ────────────

def fast_load_rgb(path_str: str) -> np.ndarray:
    with Image.open(path_str) as img:
        img.draft("RGB", (SSIM_W, SSIM_H))
        img = img.convert("RGB").resize((SSIM_W, SSIM_H), Image.BILINEAR)
        return np.asarray(img)


def fast_load_gray(path_str: str) -> np.ndarray:
    with Image.open(path_str) as img:
        img.draft("L", (SSIM_W, SSIM_H))
        img = img.convert("L").resize((SSIM_W, SSIM_H), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def process_pair(args):
    """Worker: returns (mf_str, ssim, brightness_gap, ok, err_msg)."""
    mf_str, gt_str = args
    try:
        m = fast_load_rgb(mf_str)
        g = fast_load_rgb(gt_str)
        sv  = float(ssim_metric(m, g, data_range=255, channel_axis=2))
        gap = float(abs(np.mean(m.astype(np.float32)) -
                        np.mean(g.astype(np.float32))))
        return (mf_str, sv, gap, True, "")
    except Exception as e:
        return (mf_str, -1.0, -1.0, False, str(e))


# ── everything else guarded so Windows spawn doesn't re-run it ───────────────

if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── collect pairs ─────────────────────────────────────────────────────────

    def collect_pairs():
        pairs = []
        for folder in sorted(DATASET_ROOT.iterdir()):
            if not folder.is_dir():
                continue
            for mf in sorted(folder.glob("*_moire.jpg")):
                stem = mf.stem.replace("_moire", "")
                gt   = folder / f"{stem}_gt.jpg"
                pairs.append((str(mf), str(gt)))
        return pairs

    all_pairs = collect_pairs()
    n_folders = sum(1 for p in DATASET_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(all_pairs)} candidate pairs across {n_folders} folders.")
    print(f"Using {N_WORKERS} worker processes.")

    # ── CHECK 5 — File Integrity ──────────────────────────────────────────────

    print("\n[CHECK 5] File integrity scan…")
    valid_pairs    = []
    corrupted      = []
    orphaned_moire = []
    orphaned_gt    = []

    for mf_str, gt_str in tqdm(all_pairs, unit="pair", dynamic_ncols=True):
        mf, gt = Path(mf_str), Path(gt_str)
        if not gt.exists():
            orphaned_moire.append(mf_str)
            continue
        try:
            with Image.open(mf_str) as im:
                im.verify()
            with Image.open(gt_str) as ig:
                ig.verify()
            valid_pairs.append((mf_str, gt_str))
        except Exception as e:
            corrupted.append(f"{mf.name}: {e}")

    all_gt_in_dataset = {str(p) for p in DATASET_ROOT.rglob("*_gt.jpg")}
    for gt_str in all_gt_in_dataset:
        stem  = Path(gt_str).stem.replace("_gt", "")
        moire = str(Path(gt_str).parent / f"{stem}_moire.jpg")
        if not Path(moire).exists():
            orphaned_gt.append(gt_str)

    print(f"  Valid pairs    : {len(valid_pairs)}")
    print(f"  Corrupted      : {len(corrupted)}")
    print(f"  Orphaned moiré : {len(orphaned_moire)}")
    print(f"  Orphaned GT    : {len(orphaned_gt)}")

    # ── CHECK 1 + 4 — SSIM + brightness (parallel) ───────────────────────────

    print(f"\n[CHECK 1+4] SSIM + brightness on {len(valid_pairs)} pairs "
          f"(draft @ {SSIM_W}×{SSIM_H}, {N_WORKERS} workers)…")

    ssim_scores   = {}
    exposure_gaps = {}

    with Pool(processes=N_WORKERS) as pool:
        for mf_str, sv, gap, ok, err in tqdm(
            pool.imap(process_pair, valid_pairs, chunksize=16),
            total=len(valid_pairs), unit="pair", dynamic_ncols=True
        ):
            if ok:
                ssim_scores[mf_str]   = sv
                exposure_gaps[mf_str] = gap
            else:
                corrupted.append(f"{Path(mf_str).name} (SSIM): {err}")

    # ── CHECK 2 — Phase-correlation shift (low-SSIM pairs only) ──────────────

    low_ssim_pairs = [(mf, gt) for mf, gt in valid_pairs
                      if ssim_scores.get(mf, 1.0) < SSIM_THRESH_GD]

    print(f"\n[CHECK 2] Phase-correlation on {len(low_ssim_pairs)} low-SSIM pairs…")

    shift_info = {}
    for mf_str, gt_str in tqdm(low_ssim_pairs, unit="pair", dynamic_ncols=True):
        try:
            shift, _, _ = phase_cross_correlation(
                fast_load_gray(gt_str), fast_load_gray(mf_str),
                normalization=None
            )
            shift_info[mf_str] = (float(shift[0]), float(shift[1]))
        except Exception:
            shift_info[mf_str] = (0.0, 0.0)

    # ── CHECK 3 — GT contamination (FFT) ─────────────────────────────────────

    print(f"\n[CHECK 3] GT contamination check on {GT_SAMPLE_N} random GT images…")

    all_gt_files = [gt for _, gt in valid_pairs]
    gt_sample    = random.sample(all_gt_files, min(GT_SAMPLE_N, len(all_gt_files)))
    contaminated = []
    peak_ratios  = []

    for gt_str in tqdm(gt_sample, unit="img", dynamic_ncols=True):
        try:
            with Image.open(gt_str) as img:
                gray = np.asarray(img.convert("L"), dtype=np.float32)
            H, W     = gray.shape
            spec     = np.abs(scipy_fft.fftshift(scipy_fft.fft2(gray)))
            log_spec = np.log1p(spec)
            cy, cx   = H // 2, W // 2
            Y, X     = np.ogrid[:H, :W]
            dist     = np.sqrt((Y - cy)**2 + (X - cx)**2)
            mask     = (dist > min(H, W) * 0.05) & (dist < min(H, W) * 0.49)
            vals     = log_spec[mask]
            med      = float(np.median(vals))
            p99      = float(np.percentile(vals, 99.0))
            ratio    = p99 / med if med > 0 else 0.0
            peak_ratios.append(ratio)
            if ratio > PEAK_RATIO_BAD:
                contaminated.append(
                    f"{Path(gt_str).parent.name}/{Path(gt_str).name}  ratio={ratio:.3f}"
                )
        except Exception:
            pass

    # ── Categorise ────────────────────────────────────────────────────────────

    excellent, good, warning, bad = [], [], [], []
    exposure_flagged = []
    misaligned       = []
    content_mismatch = []

    for mf_str, gt_str in valid_pairs:
        sv = ssim_scores.get(mf_str)
        if sv is None:
            continue
        mf, gt   = Path(mf_str), Path(gt_str)
        pair_str = f"{mf.parent.name}/{mf.name} | {gt.name}  SSIM={sv:.4f}"

        if sv >= SSIM_THRESH_EX:
            excellent.append(pair_str)
        elif sv >= SSIM_THRESH_GD:
            good.append(pair_str)
        elif sv >= SSIM_THRESH_WN:
            warning.append(pair_str)
        else:
            bad.append(pair_str)

        if mf_str in shift_info:
            dy, dx    = shift_info[mf_str]
            magnitude = max(abs(dy), abs(dx))
            tag       = (f"{mf.parent.name}/{mf.name}  SSIM={sv:.3f}  "
                         f"shift=({dy:.1f},{dx:.1f})px")
            if magnitude > SHIFT_THRESH:
                misaligned.append(tag)
            else:
                content_mismatch.append(tag)

        gap = exposure_gaps.get(mf_str, 0.0)
        if gap > BRIGHTNESS_GAP:
            exposure_flagged.append(
                f"{mf.parent.name}/{mf.name} | {gt.name}  brightness_gap={gap:.1f}"
            )

    # ── Save output files ─────────────────────────────────────────────────────

    def save_list(path_str, lines, header):
        p = Path(path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"# {header}\n# Generated by dataset_analysis_full.py\n\n")
            f.write("\n".join(lines) + "\n")
        print(f"  Saved {len(lines):>5} entries  →  {p}")

    print("\nSaving output files…")
    save_list("D:/pairs_excellent.txt", excellent, "SSIM >= 0.75 — safe to train on")
    save_list("D:/pairs_good.txt",      good,      "SSIM 0.60-0.75 — acceptable")
    save_list("D:/pairs_warning.txt",   warning,   "SSIM 0.40-0.60 — use with caution")
    save_list("D:/pairs_bad.txt",       bad,       "SSIM < 0.40 — remove from training")
    save_list("D:/pairs_exposure.txt",  exposure_flagged,
              "Brightness gap > 30 — remove from training")

    # ── SSIM histogram ────────────────────────────────────────────────────────

    all_ssim = list(ssim_scores.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_ssim, bins=60, color="#4C9BE8", edgecolor="white", linewidth=0.4)
    for thresh, color, label in [
        (SSIM_THRESH_EX, "#2ECC71", f"Excellent >={SSIM_THRESH_EX}"),
        (SSIM_THRESH_GD, "#F39C12", f"Good >={SSIM_THRESH_GD}"),
        (SSIM_THRESH_WN, "#E74C3C", f"Warning >={SSIM_THRESH_WN}"),
    ]:
        ax.axvline(thresh, color=color, linewidth=1.8, linestyle="--", label=label)
    ax.set_xlabel("SSIM (higher = better alignment)", fontsize=12)
    ax.set_ylabel("Number of pairs", fontsize=12)
    ax.set_title(
        f"SSIM Distribution — Full Training Set ({len(all_ssim)} pairs)", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = RESULTS_DIR / "dataset_quality.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved  →  {hist_path}")

    # ── Final Report ──────────────────────────────────────────────────────────

    total  = len(valid_pairs)
    n_ex   = len(excellent)
    n_gd   = len(good)
    n_wn   = len(warning)
    n_bd   = len(bad)
    n_exp  = len(exposure_flagged)
    n_mis  = len(misaligned)
    n_cnt  = len(content_mismatch)
    n_cont = len(contaminated)
    n_corr = len(corrupted)
    usable = n_ex + n_gd
    to_rem = n_bd + n_exp

    arr = np.array(all_ssim) if all_ssim else np.array([0.0])
    s_min  = float(arr.min())
    s_max  = float(arr.max())
    s_mean = float(arr.mean())
    s_med  = float(np.median(arr))

    pct = lambda n: f"{n / max(total, 1) * 100:.1f}%"

    if usable / max(total, 1) < 0.60 or n_bd / max(total, 1) > 0.15:
        rec = "CRITICAL"
    elif n_wn / max(total, 1) > 0.25 or n_exp / max(total, 1) > 0.10:
        rec = "WARNING"
    else:
        rec = "GOOD"

    print("""
================================================
  COMPREHENSIVE DATASET QUALITY REPORT
================================================""")
    print(f"Total pairs scanned: {total}\n")
    print("ALIGNMENT QUALITY:")
    print(f"  Excellent (>=0.75):   {n_ex:>5}  ({pct(n_ex)})")
    print(f"  Good (0.60-0.75):     {n_gd:>5}  ({pct(n_gd)})")
    print(f"  Warning (0.40-0.60):  {n_wn:>5}  ({pct(n_wn)})")
    print(f"  Bad (<0.40):          {n_bd:>5}  ({pct(n_bd)})")
    print("\nADDITIONAL ISSUES:")
    print(f"  Misaligned (shift >20px):  {n_mis:>5}")
    print(f"  Content mismatch:          {n_cnt:>5}")
    print(f"  GT contaminated:           {n_cont:>5}  (out of {GT_SAMPLE_N} sampled)")
    print(f"  Exposure mismatch:         {n_exp:>5}")
    print(f"  Corrupted files:           {n_corr:>5}")
    print("\nSSIM STATISTICS:")
    print(f"  Min / Max / Mean / Median: "
          f"{s_min:.3f} / {s_max:.3f} / {s_mean:.3f} / {s_med:.3f}")
    print(f"\nUSABLE PAIRS FOR TRAINING: {usable}  (Excellent + Good)  — {pct(usable)}")
    print(f"PAIRS TO REMOVE:           {to_rem}  (Bad + Exposure mismatch)")
    print(f"\nRECOMMENDATION: {rec}")
    if rec == "CRITICAL":
        print("  Training on this dataset will not exceed ~20 dB regardless of architecture.")
        print("  Filter to Excellent+Good pairs before training.")
    elif rec == "WARNING":
        print("  Dataset is usable but flagged pairs will cap achievable PSNR.")
        print("  Remove Bad + Exposure pairs; re-evaluate Warning pairs.")
    else:
        print("  Dataset looks healthy. Proceed with training on Excellent+Good pairs.")
    print("================================================\n")
