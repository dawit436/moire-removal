# Kaggle FHDMi Training Workflow

This is the intended flow:

1. Local machine: prepare FHDMi and upload it to Kaggle as a ZIP-backed Dataset.
2. Kaggle: start a free GPU run.
3. Kaggle: pull the relevant project files from Git.
4. Kaggle: train MBCNN on Kaggle hardware and save outputs to `/kaggle/working`.

The local computer does not train the model.

## 1. Prepare the FHDMi ZIP locally

Use the already organized local dataset if it exists at `D:/FHDMi/data`:

```powershell
python prepare_fhdmi_kaggle.py --raw-root D:/FHDMi/data --out-root D:/FHDMi_kaggle/data --budget-gb 19.0 --train-ratio 0.90
```

Create a ZIP for upload:

```powershell
Compress-Archive -Path D:/FHDMi_kaggle/data -DestinationPath D:/FHDMi_kaggle/fhdmi-kaggle.zip -Force
```

Expected layout inside the ZIP:

```text
data/
  train/
    moire/
    clean/
  test/
    moire/
    clean/
```

If your ZIP starts directly with `train/` instead of `data/train/`, the Kaggle
bootstrap can also handle that.

## 2. Upload the ZIP to Kaggle Dataset

Option A: upload `D:/FHDMi_kaggle/fhdmi-kaggle.zip` through the Kaggle web UI.

Option B: use the Kaggle CLI:

```powershell
python -m pip install --upgrade kaggle
kaggle auth login
```

```powershell
Copy-Item kaggle/dataset-metadata.template.json D:/FHDMi_kaggle/dataset-metadata.json
notepad D:/FHDMi_kaggle/dataset-metadata.json
```

Replace `YOUR_KAGGLE_USERNAME`, then:

```powershell
kaggle datasets create -p D:/FHDMi_kaggle -r zip -t
kaggle datasets status YOUR_KAGGLE_USERNAME/fhdmi-kaggle
```

For later dataset updates:

```powershell
kaggle datasets version -p D:/FHDMi_kaggle -m "Refresh FHDMi ZIP" -r zip -t
```

## 3. Configure the Kaggle Git bootstrap

Edit [kaggle/run_from_git.py](kaggle/run_from_git.py):

```python
GIT_REPO_URL = os.environ.get("GIT_REPO_URL", "https://github.com/dawit436/moire-removal.git")
GIT_BRANCH = os.environ.get("GIT_BRANCH", "master")
```

For a private repo, do not hard-code a token. In a Kaggle Notebook, use a Kaggle
Secret, then set:

```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["GIT_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")
```

The bootstrap will clone/pull the repo into:

```text
/kaggle/working/moire_project
```

Then it runs:

```text
train_combined.py --dataset fhdmi --fhdmi-data <detected Kaggle data root>
```

## 4. Create/run the Kaggle Kernel

Copy and edit metadata:

```powershell
Copy-Item kaggle/kernel-metadata.template.json kaggle/kernel-metadata.json
notepad kaggle/kernel-metadata.json
```

Replace:

```text
YOUR_KAGGLE_USERNAME/mbcnn-fhdmi-train
YOUR_KAGGLE_USERNAME/fhdmi-kaggle
```

Important: `enable_internet` must stay `true`, otherwise Kaggle cannot pull from Git.

Push the small bootstrap kernel:

```powershell
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4 --timeout 43200
kaggle kernels status YOUR_KAGGLE_USERNAME/mbcnn-fhdmi-train
```

## 5. Download trained model outputs

After the run finishes:

```powershell
kaggle kernels output YOUR_KAGGLE_USERNAME/mbcnn-fhdmi-train -p output/kaggle_fhdmi -o
```

Expected outputs:

```text
checkpoints/best_mbcnn_fhdmi.pth
best_mbcnn_fhdmi.pth
training_summary.json
```

## 6. TIP2018 warm start

If you want to fine-tune from the successful TIP2018 legacy checkpoint:

1. Upload the `.pth` file as a separate Kaggle Dataset.
2. Add that dataset id to `dataset_sources` in `kaggle/kernel-metadata.json`.
3. If there is exactly one `.pth` under `/kaggle/input`, the bootstrap passes it
   to `train_combined.py --pretrained`.

If there are multiple `.pth` files, set `PRETRAINED_CKPT` inside the Kaggle
Notebook/script environment.

## 7. Good first smoke test

Before a long free-GPU run, set smaller values in the Kaggle environment:

```python
import os
os.environ["EPOCHS"] = "2"
os.environ["BATCH_SIZE"] = "1"
os.environ["ACCUM_STEPS"] = "1"
os.environ["LR"] = "5e-5"
os.environ["L1_WEIGHT"] = "0.75"
os.environ["SSIM_WEIGHT"] = "0.20"
os.environ["FFT_WEIGHT"] = "0.05"
os.environ["SCALE_JITTER"] = "0"
```

Then run the bootstrap once. After it passes, restore the longer settings.

Recommended full FHDMi run:

```python
import os
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
