"""
Kaggle bootstrap for the real training flow:
1. FHDMi is attached to the Kaggle notebook/kernel as a Kaggle Dataset.
2. This script pulls the project from Git.
3. Training runs on Kaggle GPU, not on the local machine.

Before running on Kaggle, set GIT_REPO_URL below or as an environment variable.
For private GitHub repos, use a Kaggle Secret and set GIT_TOKEN in the notebook.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path


GIT_REPO_URL = os.environ.get("GIT_REPO_URL", "https://github.com/dawit436/moire-removal.git")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "master")
GIT_TOKEN = os.environ.get("GIT_TOKEN", "")
REPO_DIR = Path(os.environ.get("REPO_DIR", "/kaggle/working/moire_project"))

EPOCHS = os.environ.get("EPOCHS", "30")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "1")
ACCUM_STEPS = os.environ.get("ACCUM_STEPS", "8")
CROP_SIZE = os.environ.get("CROP_SIZE", "512")
LR = os.environ.get("LR", "5e-5")
NUM_WORKERS = os.environ.get("NUM_WORKERS", "2")
L1_WEIGHT = os.environ.get("L1_WEIGHT", "0.75")
SSIM_WEIGHT = os.environ.get("SSIM_WEIGHT", "0.20")
FFT_WEIGHT = os.environ.get("FFT_WEIGHT", "0.05")
SAVE_EVERY = os.environ.get("SAVE_EVERY", "5")
USE_AMP = os.environ.get("USE_AMP", "1") != "0"
SCALE_JITTER = os.environ.get("SCALE_JITTER", "0") == "1"
INSTALL_REQUIREMENTS = os.environ.get("INSTALL_REQUIREMENTS", "1") != "0"
PRETRAINED_CKPT = os.environ.get("PRETRAINED_CKPT", "")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
EXTRACT_ROOT = KAGGLE_WORKING / "fhdmi_unzipped"
CHECKPOINT_DIR = KAGGLE_WORKING / "checkpoints"


def display_cmd(cmd):
    shown = list(cmd)
    if GIT_TOKEN:
        shown = [part.replace(GIT_TOKEN, "***") for part in shown]
    return " ".join(shown)


def run(cmd, cwd=None):
    print(f"+ {display_cmd(cmd)}")
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def authenticated_url(url):
    if not GIT_TOKEN:
        return url
    if url.startswith("https://github.com/"):
        return url.replace("https://", f"https://{GIT_TOKEN}@", 1)
    return url


def clone_or_update_repo():
    repo_url = authenticated_url(GIT_REPO_URL)
    if (REPO_DIR / ".git").exists():
        run(["git", "fetch", "origin", GIT_BRANCH], cwd=REPO_DIR)
        run(["git", "checkout", GIT_BRANCH], cwd=REPO_DIR)
        run(["git", "pull", "--ff-only", "origin", GIT_BRANCH], cwd=REPO_DIR)
        return

    run([
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        GIT_BRANCH,
        repo_url,
        str(REPO_DIR),
    ])


def has_pairs(root):
    return (root / "train" / "moire").is_dir() and (root / "train" / "clean").is_dir()


def find_paired_root(search_root):
    if not search_root.exists():
        return None

    candidates = [search_root, search_root / "data"]
    candidates.extend(train_dir.parent for train_dir in search_root.rglob("train"))

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if has_pairs(resolved):
            return resolved
    return None


def safe_extract_zip(zip_path, dest):
    dest.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest.resolve()
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(dest)


def find_or_extract_fhdmi_root():
    env_root = os.environ.get("FHDMI_DATA_ROOT", "")
    if env_root:
        root = Path(env_root)
        if has_pairs(root):
            return root
        raise FileNotFoundError(f"FHDMI_DATA_ROOT is not a paired FHDMi root: {root}")

    root = find_paired_root(KAGGLE_INPUT)
    if root is not None:
        return root

    zip_files = sorted(KAGGLE_INPUT.rglob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            "No FHDMi paired folders or .zip files found under /kaggle/input."
        )

    for zip_path in zip_files:
        dest = EXTRACT_ROOT / zip_path.stem
        if not dest.exists():
            safe_extract_zip(zip_path, dest)
        root = find_paired_root(dest)
        if root is not None:
            return root

    raise FileNotFoundError(
        "Extracted zip files, but did not find train/moire and train/clean."
    )


def install_requirements():
    req = REPO_DIR / "requirements.txt"
    if INSTALL_REQUIREMENTS and req.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


def auto_pretrained_arg():
    if PRETRAINED_CKPT.lower() == "none":
        return ["--pretrained", "none"]
    if PRETRAINED_CKPT:
        return ["--pretrained", PRETRAINED_CKPT]

    checkpoints = sorted(KAGGLE_INPUT.rglob("*.pth"))
    if len(checkpoints) == 1:
        return ["--pretrained", str(checkpoints[0])]
    if len(checkpoints) > 1:
        print("Multiple .pth files found. Set PRETRAINED_CKPT to choose one:")
        for path in checkpoints:
            print(f"  {path}")
    return ["--pretrained", "none"]


def main():
    clone_or_update_repo()
    install_requirements()
    data_root = find_or_extract_fhdmi_root()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    train_script = REPO_DIR / "train_combined.py"
    if not train_script.exists():
        raise FileNotFoundError(f"train_combined.py not found in repo: {REPO_DIR}")

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset",
        "fhdmi",
        "--fhdmi-data",
        str(data_root),
        "--out-dir",
        str(CHECKPOINT_DIR),
        "--epochs",
        EPOCHS,
        "--batch-size",
        BATCH_SIZE,
        "--accum-steps",
        ACCUM_STEPS,
        "--crop-size",
        CROP_SIZE,
        "--lr",
        LR,
        "--l1-weight",
        L1_WEIGHT,
        "--ssim-weight",
        SSIM_WEIGHT,
        "--fft-weight",
        FFT_WEIGHT,
        "--num-workers",
        NUM_WORKERS,
        "--save-every",
        SAVE_EVERY,
    ]
    cmd.extend(auto_pretrained_arg())
    if USE_AMP:
        cmd.append("--amp")
    if SCALE_JITTER:
        cmd.append("--scale-jitter")

    print(f"FHDMi data root: {data_root}")
    print(f"Repo dir       : {REPO_DIR}")
    print(f"Output dir     : {CHECKPOINT_DIR}")
    run(cmd, cwd=REPO_DIR)


if __name__ == "__main__":
    main()
