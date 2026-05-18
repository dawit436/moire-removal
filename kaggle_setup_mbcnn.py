"""
kaggle_setup_mbcnn.py — Run as the first cell of a Kaggle notebook before
fine-tuning MBCNN.

Clones the latest code from GitHub (branch: master), verifies that the MBCNN
architecture and fine-tuning script are present, installs pytorch-msssim,
then prints a PASS / FAIL verdict and the command to start training.
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
    print(f"  Directory '{REPO_DIR}' already exists — pulling latest changes.")
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING — git pull failed (using cached copy):\n{result.stderr}")
    else:
        print(f"  {result.stdout.strip() or 'Already up to date.'}")
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
# Step 2 — Verify MBCNN files
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2 — Verifying MBCNN files")
print("=" * 60)

all_passed = True


def verify_file(label: str, filepath, must_contain=None):
    global all_passed
    p = Path(filepath)
    if not p.exists():
        print(f"  FAIL  [{label}] — file not found: {filepath}")
        all_passed = False
        return
    if must_contain:
        text = p.read_text(encoding="utf-8")
        for token in must_contain:
            if token not in text:
                print(f"  FAIL  [{label}] — '{token}' not found in {filepath}")
                all_passed = False
                return
    print(f"  PASS  [{label}]")


verify_file(
    "MBCNN architecture (models/mbcnn.py)",
    REPO_DIR / "models/mbcnn.py",
    must_contain=["class MBCNN", "LearnedDecomposition", "FrequencyBranch"],
)

verify_file(
    "Fine-tuning script (finetune.py)",
    REPO_DIR / "finetune.py",
    must_contain=["MBCNN", "LR", "EPOCHS"],
)

verify_file(
    "Dataset loader (dataset.py)",
    REPO_DIR / "dataset.py",
    must_contain=["MoireDataset"],
)

# ---------------------------------------------------------------------------
# Step 3 — Confirm DATA_DIR points to the Kaggle dataset
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3 — Checking DATA_DIR in finetune.py")
print("=" * 60)

kaggle_dataset_path = "moire-clean-dataset3/data"
finetune_text = (REPO_DIR / "finetune.py").read_text(encoding="utf-8")

if kaggle_dataset_path in finetune_text:
    print("  PASS  [DATA_DIR] — Kaggle dataset path found in finetune.py")
    print("         → /kaggle/input/datasets/dawitesubalew/moire-clean-dataset3/data")
else:
    print("  FAIL  [DATA_DIR] — Kaggle dataset path missing from finetune.py")
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
    print("\nReady to fine-tune. Run:")
    print("  !cd moire-removal && python finetune.py")
else:
    print("ONE OR MORE CHECKS FAILED — fix issues above before training.")
    print("=" * 60)
