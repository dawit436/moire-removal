"""
download_fhdmi.py — Download FHDMi dataset file-by-file, skipping completed ones.

Usage:
    python download_fhdmi.py

Re-run anytime to resume from where you left off.
All 30 split archives will be downloaded to D:\FHDMi_raw\

File IDs scraped from:
  https://drive.google.com/drive/folders/1IJSeBXepXFpNAvL5OyZ2Y1yu4KPvDxN5
"""

import subprocess
import sys
from pathlib import Path

OUT = Path("D:/FHDMi_raw")

# (output_relative_path, google_drive_file_id)
FILES = [
    # ── test / source ──────────────────────────────────────────────────────────
    ("test/source.tar.gz00", "1u9bX2vAzVCMXN6srr337zxWQdifrRDdM"),
    ("test/source.tar.gz01", "1JfNj5cmLmtPKKje3eS5PQPerzt0NukQ2"),
    ("test/source.tar.gz02", "1p1T0Uq3ptvLYz5mExVj6hKdw8fq_2n0D"),
    # ── test / target ──────────────────────────────────────────────────────────
    ("test/target.tar.gz00", "1M7b4oW7pTcqTFr659Ohy2CqX-FPolDN3"),
    ("test/target.tar.gz01", "1rX7009nWsTWcPRtqeZzbY8vc4H0HrE13"),
    ("test/target.tar.gz02", "1B--NQFwwHrB34MB9dcj4JQ_jpb55N6FL"),
    # ── train / source ─────────────────────────────────────────────────────────
    ("train/source.tar.gz00", "1MfrtYgiyIGydMwrdFOhioggcaYl2EUqC"),
    ("train/source.tar.gz01", "1DMUUysj9vmjsrSm8p69YMvvknREQxf-Q"),
    ("train/source.tar.gz02", "1Q6_t_U23MV1lmEpAYFBxEckuqtGMsui3"),
    ("train/source.tar.gz03", "1jSaCmmci5magmhgpy2Bsa0y25T35WSYR"),
    ("train/source.tar.gz04", "1PZSzfBCs7cYJ3lyPSz__-4OjMtmIYujd"),
    ("train/source.tar.gz05", "1UPw5ur9j6jbEElybA8vP8i984TLVPJQw"),
    ("train/source.tar.gz06", "1_gORIE3ntZjH9Nxpzmr2NrPRCtZj5Odo"),
    ("train/source.tar.gz07", "1BSkAbO2v9WKJndNgnG5SpZEsUOSlhopU"),
    ("train/source.tar.gz08", "13-UfvCeSouvaE4O-5CJe6NLPjkRFCQR1"),
    ("train/source.tar.gz09", "1sAwMtyHXgTpCz-nHmuq2B_03APo8-vAs"),
    ("train/source.tar.gz10", "1RmT6G89cGTa7SoChQkH1EU3k3_SJ3o2R"),
    ("train/source.tar.gz11", "1U_vop6HAVZPwDxuvIIhBnqnbuSpl-80F"),
    ("train/source.tar.gz12", "1qYKoIAVkAmhMeYDDjvn6H7zLVsboRYqv"),
    # ── train / target ─────────────────────────────────────────────────────────
    ("train/target.tar.gz00", "1O25WhKSOzVtZUOjKnrTH3d2gNVqGT16b"),
    ("train/target.tar.gz01", "1ndshgCYlO10EkmF81BF2amPmFBIJwz7P"),
    ("train/target.tar.gz02", "1_BPp_8YOCqN3mMf4YQJaJGOytQ5PDrJ0"),
    ("train/target.tar.gz03", "1dbaHq94hH_S9qcuvorZwkETOcMLYCV80"),
    ("train/target.tar.gz04", "1pImbiUZtxhKXnZMverlYRmSWk-zCEMVA"),
    ("train/target.tar.gz05", "1dIiFD7KTr9-soNDvraVYctvcCWV-dDuL"),
    ("train/target.tar.gz06", "13FYgTQQ1pUcVw9iJVmmN-G_lwEqJyVeN"),
    ("train/target.tar.gz07", "1YwcUG1Ii2In2Kio8gP-sXwDaImgWJVwg"),
    ("train/target.tar.gz08", "1BGvFXF6Wf6Vonmh_5GGZ6NTJcVUIYThI"),
    ("train/target.tar.gz09", "1Z2PL_ncgMiVZWjFaavPKaQkWzmWnxE_t"),
    ("train/target.tar.gz10", "1ZAFSJrbLdEuHZ__RUB4hR_Mu5bNPPHQv"),
]

# Minimum file size to consider "complete" (1.5 GB — all parts are ~2 GB)
MIN_COMPLETE_BYTES = 1_500_000_000


def is_complete(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_COMPLETE_BYTES


def main():
    pending = [(rel, fid) for rel, fid in FILES if not is_complete(OUT / rel)]
    done    = len(FILES) - len(pending)

    print(f"FHDMi download: {done}/{len(FILES)} files already complete")
    if not pending:
        print("All files downloaded!")
        return

    print(f"Downloading {len(pending)} remaining files...\n")

    for i, (rel, fid) in enumerate(pending, 1):
        out_path = OUT / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(pending)}] {rel}")

        # Remove any stale .part files for this target
        for stale in out_path.parent.glob(out_path.name + "*.part"):
            stale.unlink(missing_ok=True)

        url = f"https://drive.google.com/uc?id={fid}"
        rc = subprocess.call([
            sys.executable, "-m", "gdown",
            fid,          # pass bare file ID — gdown handles it directly
            "-O", str(out_path),
        ])
        if rc != 0:
            print(f"  [ERROR] gdown returned {rc} for {rel} — re-run to retry")
            sys.exit(1)

        size_gb = out_path.stat().st_size / 1e9
        print(f"  Done: {size_gb:.2f} GB\n")

    print(f"\nAll {len(FILES)} FHDMi files downloaded to {OUT}")


if __name__ == "__main__":
    main()
