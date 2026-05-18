# Moiré Removal Model: PSNR Ceiling Analysis

**Model:** Attention U-Net (~13.7 M params)  
**Observed ceiling:** ~20 dB PSNR  
**Target:** >25 dB  
**Files examined:** `train.py`, `dataset.py`, `models/unet.py`

---

## 1. Architecture Limitations

### 1.1 No Global Residual Connection — CRITICAL

This is the single most important flaw. The model learns a full **direct mapping** from moiré input to clean output:

```
output = Sigmoid(head(decoder_features))
```

The correct approach for any image restoration network is **residual learning**:

```
output = clamp(input + head(decoder_features), 0, 1)
```

The difference matters because the moiré artifact is a small perturbation on top of a natural image. With a direct mapping, the model must reproduce every pixel of the clean image from scratch — an extremely hard learning problem. With a residual connection, it only needs to learn **what to subtract** (the moiré noise), which has much smaller magnitude and variance. Networks like DnCNN, RRDB, and NAFNet all use this approach, and it typically yields +3 to +5 dB on its own.

The Sigmoid activation on the head compounds this. Sigmoid saturates when values approach 0 or 1, making gradients nearly zero for bright/dark pixels that are already close to correct. This creates a dead-zone in gradient flow that prevents fine-grained corrections.

**Fix:** Remove Sigmoid from the output head. Make the head output an unconstrained residual, then add the input and clamp:

```python
# In UNet.forward():
residual = self.head(x)           # no activation
return torch.clamp(x_input + residual, 0.0, 1.0)
```

The `self.head` becomes just `nn.Conv2d(base_ch // 2, out_channels, kernel_size=1)` with no Sigmoid.

### 1.2 MaxPool Downsampling

All four encoder stages use `MaxPool2d(2, 2)`. MaxPool keeps only the maximum activation in each 2×2 block and discards the other three values. This is appropriate for classification but is **lossy for reconstruction**. It causes two problems:

- Precise spatial locations of moiré features are lost in the encoder
- Strided convolutions preserve positional information better and allow the network to learn how to downsample

This is a medium-priority change; the receptive field and skip connections partially mitigate it.

### 1.3 Bottleneck Capacity Limitation

The bottleneck is `ConvBlock(512, 512)` operating at 32×32 (for 512×512 input). The receptive field of the bottleneck after 4 MaxPool operations covers 32×32 input pixels at the bottleneck level — but the effective receptive field in the original image is much larger due to skip connections. However, for moiré patterns that have large-scale interference structure, a 5th downsampling stage (to 16×16, 1024 channels) would give the bottleneck a broader view. This is a lower-priority change because it would push the model to ~50 M params and risk VRAM limits.

### 1.4 Attention Gate Asymmetry (Minor)

In `AttentionGate`, the skip-connection branch (`W_x`) applies BatchNorm but the gating signal branch (`W_g`) does not. This means the two signals being summed are at different scales after normalization. The gating signal dominates when its raw magnitude is large early in training, potentially weakening the attention mechanism. This is a minor issue that resolves itself during training but slows convergence.

---

## 2. Loss Function

### 2.1 FFT Loss Without Log-Scaling — HIGH IMPACT

The current FFT loss:

```python
pred_mag   = torch.abs(torch.fft.fft2(pred))
target_mag = torch.abs(torch.fft.fft2(target))
return F.l1_loss(pred_mag, target_mag)
```

The FFT magnitude spectrum is not log-scaled. This is a fundamental problem. In any natural image, the DC component (0,0 frequency) has a magnitude 100–1000× larger than the high-frequency components. The L1 loss on raw FFT magnitudes is therefore almost entirely dominated by **low-frequency error** — overall brightness and large structures — which is already well-handled by the L1 pixel loss.

Moiré is a **high-frequency** periodic artifact. The FFT loss, as implemented, contributes essentially zero gradient signal toward suppressing it. The 15% weight on FFT loss is largely wasted.

**Fix:** Apply log-compression before computing the loss:

```python
def fft_loss(pred, target):
    pred_mag   = torch.log1p(torch.abs(torch.fft.fft2(pred)))
    target_mag = torch.log1p(torch.abs(torch.fft.fft2(target)))
    return F.l1_loss(pred_mag, target_mag)
```

`log1p` compresses large magnitudes and amplifies small ones, balancing the spectrum so high-frequency components contribute proportionally. This single line change makes the FFT loss actually target moiré suppression.

### 2.2 SSIM Weight Too Low

At 15%, SSIM contributes weak gradient signal for structural and texture recovery. SSIM is the loss term most correlated with perceptual quality and PSNR improvements beyond 22 dB. Increasing it to 25–30% and reducing L1 accordingly would provide stronger gradient signal for recovering fine detail.

Recommended weights: `L1=0.60, SSIM=0.25, FFT=0.15` (with the log-fixed FFT).

### 2.3 Missing Perceptual Loss

Beyond 24–25 dB, pixel-space losses (L1, SSIM) alone cannot drive further improvement because their gradients become too small relative to the noise in the optimization. VGG perceptual loss computes loss in **feature space**, where the gradients for textures and edges remain strong even when pixel-space errors are already small. This is the standard approach used by all state-of-the-art restoration models (ESRGAN, Real-ESRGAN, NAFNet) to exceed 25 dB on challenging benchmarks.

Adding perceptual loss is a medium-effort change (requires a frozen VGG-19 as a feature extractor) but is likely to yield +2 to +3 dB on its own after the residual connection fix.

---

## 3. Data Pipeline

### 3.1 Color Jitter Applied Only to Moiré Input — HIGH IMPACT

In `dataset.py`, `ColorJitter` (brightness ±0.2, contrast ±0.2) is applied only to the moiré image, not the clean target:

```python
moire_img = self._jitter(moire_img)   # only moiré is jittered
# clean_img is unchanged
```

This teaches the model that the input brightness and contrast **differ systematically** from the output. At test time this is false — the moiré image and clean image should have the same overall brightness. The model wastes a significant portion of its capacity learning a brightness-correction mapping that doesn't exist at test time, and it introduces ambiguity: a pixel that is darker in the moiré image might be dark because of the moiré pattern or because of the artificial jitter.

**Fix:** Either remove the color jitter entirely, or apply it identically to both images (which would still simulate illumination variation without creating a mismatch).

### 3.2 Multi-Scale Crop Resizing Corrupts Moiré Frequencies

The pipeline randomly picks a crop size from `[384, 512, 640]` and then resizes all crops to 512×512:

- 640px crop resized to 512px: downsamples by factor 0.8
- 384px crop resized to 512px: upsamples by factor 1.33

When a 640px region is downsampled to 512px, a moiré pattern at 30 cycles/image becomes 37.5 cycles/image. The model sees the **same moiré artifact at different effective spatial frequencies** depending on which scale was chosen. This undermines learning because the FFT loss (even if fixed) measures frequency content at the 512px scale, not at the native moiré frequency.

For the multi-scale augmentation to be beneficial, the resize should be removed and all crops should be used at their native size. This requires a model that accepts variable-size inputs (fully convolutional, which this U-Net already is) and a collation strategy that handles variable-size batches (e.g., only use fixed-size crops, or use padding). The simpler fix is to just remove the scale variation and always crop at 512×512.

### 3.3 Random Rotation Is Counterproductive for Moiré

Moiré patterns have **orientation-specific structure** tied to the angle between the camera sensor grid and the screen pixel grid. Rotating by 90° or 270° produces a moiré pattern oriented differently from anything the model would see at test time (where the camera-to-screen angle is random but fixed during a single photo). The 180° rotation is fine (same orientation).

Horizontal and vertical flips are fine — they don't change the fundamental moiré frequency, only mirror it.

**Fix:** Remove the `random.choice([90, 180, 270])` rotation or restrict it to 180° only.

---

## 4. Training Hyperparameters

### 4.1 No Gradient Clipping

The attention gates can produce sharp gradients, especially early in training when the attention weights are near 0 or 1. Without gradient clipping, a bad batch can cause a large update that moves weights far from a good local minimum. Adding `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()` costs nothing and prevents instability.

### 4.2 No Warmup

Starting at LR=1e-4 immediately can cause early oscillation, especially with attention gates. A 5-epoch linear warmup from 1e-6 to 1e-4 would help the model settle into a good optimization trajectory before the cosine decay begins. This is a low-effort, moderate-impact change.

### 4.3 Cosine Annealing Period

`CosineAnnealingLR(T_max=EPOCHS)` produces a single cosine cycle over all 50 epochs, ending at `eta_min=1e-6`. This is correct but aggressive — the LR drops to near-zero at epoch 50. If training continues for more epochs (which it should), the LR is too low to make meaningful updates. Increasing `T_max` to 1.5× the epoch count would keep learning rates productive longer.

### 4.4 Batch Size 4 and BatchNorm

BatchNorm statistics are computed over the spatial positions within the batch. With batch=4 and 512×512 crops, there are 4 × 512 × 512 ≈ 1 million positions per channel — more than enough for stable BatchNorm statistics. This is not a problem.

---

## 5. Dataset Size and Quality

### 5.1 Unknown but Likely Small

The Kaggle dataset name suggests a curated dataset of modest size (typically 100–500 image pairs for moiré removal datasets). If the dataset has fewer than 300 training pairs, the model will show training loss much lower than validation loss after epoch 20, which is a clear sign of overfitting. With 512×512 crops and only a few hundred images, each epoch sees approximately:

```
300 images × (4032 × 3024) / (512 × 512) ≈ 11,000 non-overlapping crop positions
```

This is not a large training set for a 13.7M parameter model. Aggressive augmentation (without the broken color jitter) is essential.

### 5.2 JPEG Compression Artifacts in Ground Truth

Both the moiré and clean images are JPEG (`.jpg`). If the clean ground truth was re-compressed with a different quality factor than the moiré images, the model cannot distinguish JPEG ringing artifacts from moiré artifacts and will average between the two, creating a hard ceiling below 30 dB regardless of architecture improvements. This is impossible to fix without the original uncompressed data.

---

## 6. Priority Rankings — Impact vs Effort

| # | Change | PSNR Gain Estimate | Effort | Ratio |
|---|--------|--------------------|--------|-------|
| 1 | Global residual connection + remove Sigmoid | +3 to +5 dB | Low (10 lines) | **Highest** |
| 2 | Fix FFT loss with log1p scaling | +1 to +2 dB | Very Low (1 line) | **Very High** |
| 3 | Remove color jitter from moire-only (apply to both or remove) | +1 to +2 dB | Very Low (1 line) | **Very High** |
| 4 | Increase SSIM weight to 0.25 | +0.5 to +1 dB | Very Low | High |
| 5 | Remove resize from multi-scale augmentation | +0.5 to +1 dB | Low | High |
| 6 | Add perceptual (VGG) loss | +2 to +3 dB | Medium (50 lines) | High |
| 7 | Add gradient clipping | stability improvement | Very Low | High |
| 8 | Add LR warmup | +0.3 to +0.5 dB | Low | Medium |
| 9 | Remove 90°/270° rotation | +0.3 dB | Very Low | Medium |
| 10 | Replace MaxPool with strided conv | +0.5 to +1 dB | Medium | Medium |

---

## 7. What Each Training Run Should Target

### Run 1 — Fix the foundations (expected result: 23–25 dB)

Make only these changes. They are all low-effort and address the root causes:

1. **Add global residual connection** in `unet.py`: remove Sigmoid, change head to a plain conv, and add `torch.clamp(x_input + residual, 0, 1)` in `forward()`.
2. **Fix FFT loss** in `train.py`: add `torch.log1p()` before computing L1 on magnitudes.
3. **Fix color jitter** in `dataset.py`: apply to both images or remove entirely.
4. **Reweight loss** in `train.py`: `L1=0.60, SSIM=0.25, FFT=0.15`.
5. **Remove resize from multi-scale crops** in `dataset.py`: after the crop, do not resize; use the sampled crop size as-is. Remove the `[384, 512, 640]` variation entirely and just always crop at 512×512.
6. **Add gradient clipping** in `train_one_epoch()`.

These six changes require modifying ~20 lines across three files. The residual connection alone should push PSNR from 20 to 23+ dB.

### Run 2 — Add perceptual signal (expected result: 25–27 dB)

After Run 1 converges:

1. **Add VGG perceptual loss** using features from VGG-19 `relu2_2` and `relu3_3`. Weight it at 0.1 of the total loss, reducing L1 to 0.50 (`L1=0.50, SSIM=0.25, FFT=0.15, Perceptual=0.10`).
2. **Add LR warmup** (5 epochs linear from 1e-6 to 1e-4).
3. **Resume from Run 1's best checkpoint** rather than training from scratch — load the weights and fine-tune with perceptual loss.

### Run 3 — Architecture upgrade (expected result: 27–30 dB)

If Run 2 plateaus:

1. **Replace MaxPool with strided Conv2d** in the encoder.
2. **Add a 5th encoder stage** (512 → 1024 channels, 16×16 spatial at 512px input) to increase the bottleneck's effective receptive field.
3. Consider replacing the base U-Net with **NAFNet** (Nonlinear Activation Free Network), which consistently achieves 30+ dB on similar tasks with comparable parameter counts.

---

## 8. Realistic Ceiling

| Scenario | Expected PSNR |
|----------|---------------|
| Current (no changes) | ~20 dB |
| After Run 1 (residual + fixed losses + augmentation) | 23–25 dB |
| After Run 2 (+ perceptual loss, fine-tuning) | 25–27 dB |
| After Run 3 (architecture upgrade) | 27–30 dB |
| Hard ceiling from JPEG dataset artifacts | ~30–32 dB |

The JPEG ceiling is real. If the dataset ground-truth images were saved as JPEG at quality 85–90, the noise floor introduced by JPEG compression corresponds to roughly 35–38 dB. Any model will struggle to exceed this because the "clean" target itself is not truly clean.

The most realistic goal for 2–3 more runs starting from the current code is **25–27 dB**, achievable without any architecture changes — just fixing the residual connection, FFT loss, color jitter, and adding perceptual loss.

---

## 9. Immediate Actions (Before the Next Run)

In order of priority, before starting the next training run:

```
1. unet.py   — Remove Sigmoid from head; add residual in forward()
2. train.py  — Fix fft_loss() with torch.log1p()
3. train.py  — Change weights: L1=0.60, SSIM=0.25, FFT=0.15
4. train.py  — Add clip_grad_norm_(model.parameters(), 1.0) before optimizer.step()
5. dataset.py — Remove color jitter (or apply to both images)
6. dataset.py — Remove multi-scale crop resizing (always use self.crop_size)
```

Items 2, 3, 4, 5, 6 take under 10 minutes total. Item 1 takes 20–30 minutes and has the highest payoff. Do all six before training.
