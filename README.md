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
| `train.py` | Training script (works on synthetic demo data out of the box, or real HAM10000) |
| `app.py` | Streamlit dashboard (upload → preprocess → predict → Grad-CAM → charts) |
| `requirements.txt` | Python dependencies |
| `checkpoints/` | Where trained model weights (`.pth`) get saved |

This project has already been smoke-tested end-to-end (dataset → preprocessing →
model forward pass → Grad-CAM → checkpoint save/reload) with both backbones,
so the code is verified to run, not just compile.

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
   `model.py`, `gradcam.py`, `train.py`, `app.py`, `requirements.txt`)
   into that folder, keeping the exact filenames (the modules import each
   other by these names).
3. Create an empty subfolder named `checkpoints/` inside it — this is
   where trained weights will be saved.

Your folder should look like this:

```
skin_lesion_app/
├── app.py
├── config.py
├── dataset.py
├── gradcam.py
├── model.py
├── preprocessing.py
├── train.py
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

## 7. (Optional) Step-by-step: train / fine-tune the model

### 7.1 Quick smoke run (synthetic data, no download — just verifies everything works)
```bash
python train.py --epochs 3 --batch_size 8 --backbone efficientnet_b4
```
This trains on procedurally generated synthetic lesion images (see
`dataset.py`) so you can confirm the training loop, GPU/CPU detection, and
checkpoint saving all work before committing to a long real-data run. It
saves the best checkpoint to `checkpoints/skin_lesion_model.pth`.

### 7.2 Real fine-tuning on HAM10000
```bash
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

Training prints per-epoch loss/accuracy and only overwrites the checkpoint
when validation accuracy improves, so `checkpoints/skin_lesion_model.pth`
always holds your best model.

### 7.3 Use your trained model in the app
Once training finishes, just run:
```bash
streamlit run app.py
```
The sidebar's **"Fine-tuned checkpoint path"** field already defaults to
`checkpoints/skin_lesion_model.pth` — the app will auto-load it (you'll see
a green "Loaded fine-tuned checkpoint" message in the sidebar). If it's
somewhere else, paste the path into that field.

---

## 8. Using the dashboard

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

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Activate your venv, then re-run `pip install -r requirements.txt` |
| `pip install` fails on torch for your platform | Install PyTorch manually first from https://pytorch.org/get-started/locally/, matching your OS/CUDA, then rerun `pip install -r requirements.txt` |
| `CUDA out of memory` during training | Lower `--batch_size` (e.g. `--batch_size 4`), or drop `--backbone` to `resnet50`, or train on CPU |
| Streamlit sidebar shows "No fine-tuned checkpoint found" | Expected until you run `train.py` — the app still works with the ImageNet-pretrained backbone, just without domain-specific accuracy |
| `FileNotFoundError: Metadata CSV not found` when using `--real_data` | Double-check `--data_dir` points at the folder containing `HAM10000_metadata.csv` directly (see the folder layout in Section 6) |
| `ValueError: Expected more than 1 value per channel when training` | Already handled in `train.py` via `drop_last=True` on the training loader — if you modify the training loop yourself, keep this to avoid BatchNorm crashing on a final batch of size 1 |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Slow first run | The very first `build_model()` call downloads ImageNet-pretrained weights (~75 MB for EfficientNet-B4); subsequent runs use the local cache in `~/.cache/torch/hub/checkpoints/` |

---

## 10. Extending this project

- Swap in `efficientnet_b0`/`b7`, or any `torchvision.models` backbone, by
  adding a branch in `SkinLesionClassifier.__init__` (`model.py`) — just
  point `target_layer_name` at that backbone's last conv layer for Grad-CAM
  to keep working.
- Add test-time augmentation (TTA) by averaging `predict()` results over
  a few augmented copies of the same image.
- Add batch/offline inference by looping `predict()` over a folder instead
  of using the Streamlit uploader.
#   S k i n - L e s i o n - a n d - M e l a n o m a - D e t e c t i o n  
 