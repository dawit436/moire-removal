"""
kaggle_setup.py — Run this as the first cell of a Kaggle notebook.

Clones the latest code from GitHub (branch: master), verifies the three
key fixes exist, confirms DATA_DIR points to the Kaggle dataset, installs
pytorch-msssim, then prints a final PASS / FAIL verdict.
"""

import subprocess
import sys
from pathlib import Path

REPO_URL    = "https://github.com/dawit436/moire-removal.git"
REPO_BRANCH = "master"
REPO_DIR    = Path("moire-removal")

# ---------------------------------------------------------------------------
# Step 1 — Clone
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1 — Cloning repository (branch: master)")
print("=" * 60)

if REPO_DIR.exists():
    print(f"  Directory '{REPO_DIR}' already exists — skipping clone.")
else:
    result = subprocess.run(
        ["git", "clone", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  FAIL — git clone failed:\n{result.stderr}")
        sys.exit(1)
    print(f"  Cloned to {REPO_DIR.resolve()}")

# ---------------------------------------------------------------------------
# Step 2 — Verify the 3 fixes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — Verifying the 3 key fixes")
print("=" * 60)

all_passed = True

def verify(label, filepath, must_contain=None, must_not_contain=None):
    global all_passed
    text = Path(filepath).read_text(encoding="utf-8")
    if must_contain:
        for token in must_contain:
            if token not in text:
                print(f"  FAIL  [{label}] — '{token}' not found in {filepath}")
                all_passed = False
                return
    if must_not_contain:
        for token in must_not_contain:
            if token in text:
                print(f"  FAIL  [{label}] — '{token}' should NOT be in {filepath}")
                all_passed = False
                return
    print(f"  PASS  [{label}]")

verify(
    "Residual connection (models/unet.py)",
    REPO_DIR / "models/unet.py",
    must_contain=["moire_input", "torch.clamp"],
)
verify(
    "log1p FFT loss (train.py)",
    REPO_DIR / "train.py",
    must_contain=["log1p"],
)
verify(
    "Color jitter removed (dataset.py)",
    REPO_DIR / "dataset.py",
    must_not_contain=["self._jitter"],
)

# ---------------------------------------------------------------------------
# Step 3 — Confirm DATA_DIR points to the Kaggle dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — Checking DATA_DIR")
print("=" * 60)

kaggle_dataset_path = "moire-clean-dataset3/data"
train_text = (REPO_DIR / "train.py").read_text(encoding="utf-8")

if kaggle_dataset_path in train_text:
    print("  PASS  [DATA_DIR] — Kaggle dataset path found in train.py")
    print("         → /kaggle/input/datasets/dawitesubalew/moire-clean-dataset3/data")
else:
    print("  FAIL  [DATA_DIR] — Kaggle dataset path missing from train.py")
    all_passed = False

# ---------------------------------------------------------------------------
# Step 4 — Install pytorch-msssim
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4 — Installing pytorch-msssim")
print("=" * 60)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pytorch-msssim"],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print("  PASS  [pytorch-msssim] — installed successfully")
else:
    print(f"  FAIL  [pytorch-msssim] — install error:\n{result.stderr}")
    all_passed = False

# ---------------------------------------------------------------------------
# Step 5 — Verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if all_passed:
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print("\nReady to train. Run:")
    print("  !cd moire-removal && python train.py")
else:
    print("ONE OR MORE CHECKS FAILED — fix issues above before training.")
    print("=" * 60)
