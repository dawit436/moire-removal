# FHDMi MBCNN Next Training Notes

## What the previous comparisons showed

`results/model_comparison_metrics.txt` used the same 30 test pairs and the same
preprocessing for every loaded model:

```text
best_model_attention [UNet]      PSNR 17.41 / SSIM 0.6608
best_mbcnn_tip2026 [MBCNN]       PSNR 15.55 / SSIM 0.5223
best_mbcnn_tip2018_50k192        PSNR 15.26 / SSIM 0.5605
best_mbcnn_tip2018 [Legacy]      PSNR 15.25 / SSIM 0.6164
```

Important: checkpoint `val_psnr` values came from different validation sets and
are not comparable. Only the shared test-pair comparison above is meaningful.

## Visual diagnosis

The MBCNN variants often correct too aggressively:

- global brightness/contrast shifts
- local detail damage on faces/textures
- moire lines still visible in hard regions
- much slower full-resolution CPU inference because all three branches run at
  full resolution

This means the next FHDMi run should not simply increase epochs. It should make
the optimization less destructive and preserve image content better.

## Changes made for the next Kaggle run

- `dataset.py` now supports `scale_jitter=False`.
- `train_combined.py` exposes `--l1-weight`, `--ssim-weight`, `--fft-weight`.
- `train_combined.py` exposes `--scale-jitter`; Kaggle defaults keep it off.
- `kaggle/run_from_git.py` defaults to a more stable FHDMi fine-tuning setup:
  - `BATCH_SIZE=1`
  - `ACCUM_STEPS=8`
  - `LR=5e-5`
  - `L1_WEIGHT=0.75`
  - `SSIM_WEIGHT=0.20`
  - `FFT_WEIGHT=0.05`
  - `SCALE_JITTER=0`
  - `USE_AMP=1`
- `models/mbcnn.py` now zero-initializes the final output conv so training from
  scratch starts as an identity-like mapping.

## Why these changes

Moire frequency depends on scale. Randomly resizing cropped patches changes the
interference frequency and can teach the model artifacts that do not match FHDMi
native images.

The old FFT weight (`0.30`) pushed the model to fight frequency energy strongly,
which can explain the over-brightening and content distortion seen in visual
comparisons. The new setting still uses FFT, but makes RGB/SSIM reconstruction
dominate the objective.

## First Kaggle check

Run a smoke test first:

```python
os.environ["EPOCHS"] = "2"
os.environ["BATCH_SIZE"] = "1"
os.environ["ACCUM_STEPS"] = "1"
```

Then run the full setup:

```python
os.environ["EPOCHS"] = "30"
os.environ["BATCH_SIZE"] = "1"
os.environ["ACCUM_STEPS"] = "8"
os.environ["CROP_SIZE"] = "512"
os.environ["LR"] = "5e-5"
os.environ["L1_WEIGHT"] = "0.75"
os.environ["SSIM_WEIGHT"] = "0.20"
os.environ["FFT_WEIGHT"] = "0.05"
os.environ["SCALE_JITTER"] = "0"
os.environ["USE_AMP"] = "1"
```
