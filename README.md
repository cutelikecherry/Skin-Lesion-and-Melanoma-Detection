# 🩺 Automated Skin Lesion Classification & Melanoma Detection System

A PyTorch + OpenCV + Streamlit application that classifies dermoscopic skin
lesion images into 7 diagnostic categories (HAM10000 taxonomy), removes
hair artifacts with the DullRazor algorithm, and explains every prediction
with a Grad-CAM heatmap.

> ⚠️ **Disclaimer:** This is a research/educational project. It is **not**
> a certified medical device and must **never** be used as a substitute
> for evaluation by a qualified dermatologist.

---

## 1. What's in this folder

| File | Purpose |
|---|---|
| `config.py` | Class names, risk-level mapping, image size, normalization constants |
| `preprocessing.py` | OpenCV DullRazor hair removal, resizing, tensor conversion |
| `dataset.py` | `HAM10000Dataset` (real data) + `SyntheticLesionDataset` (zero-download demo data), augmentations |
| `model.py` | EfficientNet-B4 / ResNet50 classifier, checkpoint I/O, inference + risk logic |
| `gradcam.py` | Grad-CAM hook-based explainability engine + heatmap overlay |
| `feature_cache.py` | Frozen-backbone feature extraction + `.pkl` cache read/write (fast CPU workflow) |
| `extract_features.py` | **Step 1 (fast CPU path):** run every image through the backbone once, cache the features |
| `train_head.py` | **Step 2 (fast CPU path):** train only the classifier head on cached features, save a lightweight `.pkl` model |
| `train.py` | Full fine-tuning script (slow on CPU — recommended only if you have a GPU) |
| `app.py` | Streamlit dashboard (upload → preprocess → predict → Grad-CAM → charts) |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Where trained model weights get saved |

This project has been smoke-tested end-to-end in a live sandbox: dataset →
preprocessing → model forward pass → Grad-CAM → checkpoint save/reload, for
**both** the full-fine-tuning path and the fast frozen-feature path below,
including timing measurements. It's verified to run, not just compile.

> **On a CPU, use Section 7 ("Fast CPU workflow"), not `train.py`.**
> `train.py` fine-tunes the *entire* EfficientNet-B4 backbone, which means
> every epoch re-runs a full forward **and backward** pass through a ~19M
> parameter CNN — that's what was taking you hours. Section 7 instead runs
> the backbone forward pass **once per image, with no backward pass at
> all**, caches the results, and trains only a small classifier head on
> those cached vectors. In sandbox testing this brought "training" down to
> low single-digit seconds for dozens of epochs, after a one-time feature
> extraction step (~0.2s/image on CPU for EfficientNet-B4 forward-only).

---

## 2. Prerequisites

- **Python 3.9 – 3.12** (project was verified on 3.12)
- **pip**
- ~3 GB free disk space (PyTorch + torchvision + pretrained weights)
- A GPU is optional — everything runs on CPU too, just slower

Check your Python version:
```bash
python3 --version
```

---

## 3. Step-by-step: download / save the files

1. Create a project folder on your machine, e.g. `skin_lesion_app/`.
2. Save each file above (`config.py`, `preprocessing.py`, `dataset.py`,
   `model.py`, `gradcam.py`, `feature_cache.py`, `extract_features.py`,
   `train_head.py`, `train.py`, `app.py`, `requirements.txt`) into that
   folder, keeping the exact filenames (the modules import each other by
   these names).
3. Create an empty subfolder named `checkpoints/` inside it — this is
   where trained weights will be saved.

Your folder should look like this:

```
skin_lesion_app/
├── app.py
├── config.py
├── dataset.py
├── extract_features.py   (fast CPU workflow: step 1)
├── feature_cache.py
├── gradcam.py
├── model.py
├── preprocessing.py
├── train_head.py         (fast CPU workflow: step 2)
├── train.py               (full fine-tuning, GPU recommended)
├── requirements.txt
└── checkpoints/          (empty for now)
```

---

## 4. Step-by-step: environment setup

Open a terminal **inside** the `skin_lesion_app/` folder.

### 4.1 Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked when your terminal prompt shows `(venv)` at the start.

### 4.2 Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `torch`, `torchvision`, `opencv-python-headless`, `streamlit`,
`plotly`, `pandas`, `numpy`, `Pillow`.

> If you have an NVIDIA GPU and want CUDA acceleration, instead install the
> matching CUDA build of PyTorch first from https://pytorch.org/get-started/locally/
> (pick your CUDA version), **then** run `pip install -r requirements.txt`
> — pip will skip reinstalling torch if it's already satisfied.

---

## 5. Step-by-step: run the app immediately (no dataset download needed)

The app works out of the box with an **ImageNet-pretrained** backbone (no
fine-tuning) — this lets you see the full UI (upload, hair removal,
Grad-CAM, charts) immediately. Predictions won't be clinically meaningful
until you fine-tune (Section 7), but every module actually runs.

```bash
streamlit run app.py
```

Streamlit will print a local URL, e.g.:
```
Local URL: http://localhost:8501
```
Open that URL in your browser. Upload any skin/lesion photo (PNG or JPG) to
see:
- Raw image vs. DullRazor hair-removed image, side by side
- Predicted class + color-coded risk banner (High/Medium/Low)
- Grad-CAM heatmap overlay showing which region drove the prediction
- Interactive Plotly bar chart of all 7 class probabilities

To stop the app, go back to the terminal and press `Ctrl+C`.

---

## 6. (Optional) Step-by-step: download the real HAM10000 dataset

Skip this section if you're just testing the pipeline with synthetic data.

1. Go to the Kaggle dataset page: **"Skin Cancer MNIST: HAM10000"**
   (search "skin-cancer-mnist-ham10000" on kaggle.com), or the official
   ISIC Archive (https://api.isic-archive.com).
2. Download and unzip it. You should end up with:
   - `HAM10000_metadata.csv`
   - A folder of `.jpg` images (Kaggle ships them split across
     `HAM10000_images_part_1/` and `_part_2/` — merge both into one folder)
3. Reorganize into the layout this project expects:
   ```
   ham10000_data/
   ├── HAM10000_metadata.csv
   └── images/
       ├── ISIC_0024306.jpg
       ├── ISIC_0024307.jpg
       └── ...
   ```
4. Place the `ham10000_data/` folder next to your `skin_lesion_app/` code
   (or anywhere — you'll pass its path with `--data_dir`).

If you use the Kaggle CLI instead of the website:
```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d ham10000_data
```
(You'll still need to merge the two image-part folders into a single
`images/` folder as shown above.)

---

## 7. Fast CPU workflow: cache features, then train only the head (recommended)

This is the two-step approach built specifically so you can train on a CPU
in seconds-to-minutes instead of hours, and then **delete the dataset**
once you have the cache.

**Why this is fast:** `train.py` re-runs the whole EfficientNet-B4 backbone
(forward *and* backward) on every epoch. Here, instead, the backbone runs
forward **exactly once per image, with no backward pass**, and the result
(a short numeric vector) is cached to disk. Training then only touches a
tiny 3-layer classifier head on those cached vectors — no images, no CNN,
no backward pass through 19M parameters — so many epochs finish in seconds.

### 7.1 Step 1 — extract and cache features (run once)

```bash
# Zero-download demo (synthetic data) — just to see the workflow run:
python extract_features.py --backbone efficientnet_b4

# Real HAM10000 data:
python extract_features.py --real_data --data_dir ./ham10000_data --backbone efficientnet_b4
```

This preprocesses every image (DullRazor hair removal + resize, same as
before), runs it through the frozen pretrained backbone, and writes
**`features_cache.pkl`** — a compact file (kilobytes to a few MB, not
gigabytes) containing the extracted feature vectors + labels for both the
train and validation splits.

Useful flags:
| Flag | Default | Meaning |
|---|---|---|
| `--backbone` | efficientnet_b4 | `efficientnet_b4` or `resnet50` — try `resnet50` if extraction is still slow on your CPU |
| `--real_data` | off | Use real HAM10000 instead of synthetic demo data |
| `--data_dir` | ./ham10000_data | Path to your downloaded dataset |
| `--batch_size` | 16 | Extraction-only batch size — safe to raise much higher than a training batch size (no gradients are stored) |
| `--num_workers` | 0 | DataLoader worker processes; try 2–4 if you have spare CPU cores |
| `--train_augment_copies` | 3 | How many independently-augmented feature vectors to cache **per training image** — adds cheap augmentation diversity at a one-time cost instead of a per-epoch cost |
| `--output` | features_cache.pkl | Output path |

### 7.2 Delete the dataset

Once `extract_features.py` finishes, you no longer need the raw images —
everything downstream reads only from `features_cache.pkl`. Safe to delete
`ham10000_data/` (or wherever you unzipped it) to free disk space.

### 7.3 Step 2 — train the classifier head (fast, repeat as often as you like)

```bash
python train_head.py --features_cache features_cache.pkl --epochs 40
```

Because this trains only a small head on already-extracted vectors, you
can freely re-run it with different `--epochs`, `--lr`, or `--batch_size`
to experiment — each run takes seconds, not hours. It saves the best
checkpoint to **`checkpoints/skin_lesion_model.pkl`**.

Useful flags:
| Flag | Default | Meaning |
|---|---|---|
| `--features_cache` | features_cache.pkl | Path to the cache from Step 1 |
| `--epochs` | 40 | Number of epochs (cheap — raise this freely) |
| `--batch_size` | 32 | Batch size for the head-only training loop |
| `--lr` | 1e-3 | Learning rate |
| `--checkpoint_out` | checkpoints/skin_lesion_model.pkl | Where to save the best model |

This `.pkl` file deliberately contains **only** the trained classifier head
plus metadata (backbone name, class names) — not the backbone's ~75–100 MB
of weights, since those are the same standard pretrained ImageNet weights
everywhere. That keeps the checkpoint small and portable: `load_lightweight_checkpoint()`
(used internally by `app.py`) rebuilds the exact same pretrained backbone
from `torchvision` on any machine and drops your trained head on top of it
— **no original dataset needed**, just this one small file plus an internet
connection the first time (to fetch the standard backbone weights, which
`torchvision` then caches locally for all future runs).

### 7.4 Use your trained model in the app

```bash
streamlit run app.py
```
The sidebar's **"Trained checkpoint path"** field already defaults to
`checkpoints/skin_lesion_model.pkl` — the app auto-detects it's a
lightweight checkpoint from the `.pkl` extension and loads it (you'll see
a green confirmation message in the sidebar naming the backbone it was
trained with). Move that one `.pkl` file to any other copy of this project
— on another machine, after wiping the dataset, wherever — and it'll work
the same way.

---

## 8. Alternative: full fine-tuning with `train.py` (GPU recommended)

If you have a GPU (or don't mind a long CPU run), `train.py` fine-tunes
the entire backbone end-to-end, which can reach higher accuracy than the
frozen-feature approach above at the cost of much more compute:

```bash
# Quick smoke run on synthetic data:
python train.py --epochs 3 --batch_size 8 --backbone efficientnet_b4

# Real fine-tuning on HAM10000:
python train.py --real_data --data_dir ./ham10000_data --epochs 15 --batch_size 16 --backbone efficientnet_b4
```

Useful flags:
| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 8 | Batch size (lower this if you hit out-of-memory) |
| `--lr` | 1e-4 | Learning rate |
| `--backbone` | efficientnet_b4 | `efficientnet_b4` or `resnet50` |
| `--real_data` | off | Use real HAM10000 instead of synthetic data |
| `--data_dir` | ./ham10000_data | Path to your downloaded dataset |
| `--checkpoint_out` | checkpoints/skin_lesion_model.pth | Where to save the best weights |

This saves a full `.pth` state-dict checkpoint (~75–100 MB, since it
includes the fine-tuned backbone). Point the app's sidebar checkpoint field
at that `.pth` path instead — `app.py` auto-detects the extension and loads
it with the full-state-dict loader onto the backbone selected in the
sidebar (rather than the `.pkl` lightweight loader used in Section 7).

---

## 9. Using the dashboard

1. **Sidebar** — choose the backbone (EfficientNet-B4 or ResNet50), point
   at a checkpoint file (optional), set the confidence threshold that
   triggers an "uncertain prediction" warning, and toggle hair removal /
   Grad-CAM on or off.
2. **Upload** a PNG/JPG dermoscopic image.
3. The app shows, top to bottom: raw vs. hair-removed image, the risk
   banner with predicted class + confidence, the Grad-CAM heatmap, and the
   probability bar chart. Expand "Full probability table" for exact numbers
   per class.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Activate your venv, then re-run `pip install -r requirements.txt` |
| `pip install` fails on torch for your platform | Install PyTorch manually first from https://pytorch.org/get-started/locally/, matching your OS/CUDA, then rerun `pip install -r requirements.txt` |
| `CUDA out of memory` during training | Lower `--batch_size` (e.g. `--batch_size 4`), or drop `--backbone` to `resnet50`, or train on CPU |
| Streamlit sidebar shows "No checkpoint found" | Expected until you run Section 7 (or `train.py`) — the app still works with the ImageNet-pretrained backbone, just without domain-specific accuracy |
| `FileNotFoundError: Metadata CSV not found` when using `--real_data` | Double-check `--data_dir` points at the folder containing `HAM10000_metadata.csv` directly (see the folder layout in Section 6) |
| `ValueError: Expected more than 1 value per channel when training` | Already handled in both `train.py` and `train_head.py` via `drop_last=True` on the training loader — if you modify either training loop yourself, keep this to avoid BatchNorm crashing on a final batch of size 1 |
| `extract_features.py` still feels slow | It should be far faster than `train.py` per image (no backward pass), but EfficientNet-B4 is still a big network on CPU. Try `--backbone resnet50` (lighter), lower `--batch_size` if you're low on RAM, or reduce `--train_augment_copies` |
| `train_head.py` raises `ValueError: Feature cache at ... is missing keys` | The `.pkl` you pointed it at wasn't produced by `extract_features.py` — re-run Step 1, or check `--features_cache` points at the right file |
| `train_head.py` finishes almost instantly with poor accuracy | Expected on the tiny synthetic demo dataset — that's just proving the pipeline runs. Point `extract_features.py --real_data` at real HAM10000 data for meaningful accuracy, and consider raising `--train_augment_copies` |
| App's sidebar checkpoint field: `.pkl` vs `.pth` | `.pkl` = lightweight head-only checkpoint from `train_head.py` (Section 7), auto-detects its backbone. `.pth` = full state-dict checkpoint from `train.py` (Section 8), loaded onto the sidebar's selected backbone. Don't mix up the extensions |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Slow first run | The very first `build_model()` call downloads standard ImageNet-pretrained backbone weights (~75 MB for EfficientNet-B4); subsequent runs use the local cache in `~/.cache/torch/hub/checkpoints/`. This download happens once regardless of which workflow you use, and is unrelated to your own dataset |

---

## 11. Extending this project

- Swap in `efficientnet_b0`/`b7`, or any `torchvision.models` backbone, by
  adding a branch in `SkinLesionClassifier.__init__` (`model.py`) — just
  point `target_layer_name` at that backbone's last conv layer for Grad-CAM
  to keep working.
- Add test-time augmentation (TTA) by averaging `predict()` results over
  a few augmented copies of the same image.
- Add batch/offline inference by looping `predict()` over a folder instead
  of using the Streamlit uploader.
