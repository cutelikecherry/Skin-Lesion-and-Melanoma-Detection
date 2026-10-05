
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
HAM10000_DX_MAP = {
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular Lesion",
}


def get_train_transforms(image_size: int = IMAGE_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = IMAGE_SIZE):
    """Deterministic validation/inference-time transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HAM10000Dataset(Dataset):

    def __init__(self, data_dir: str, metadata_csv: str = "HAM10000_metadata.csv",
                 image_subdir: str = "images", transform=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, image_subdir)
        csv_path = os.path.join(data_dir, metadata_csv)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Metadata CSV not found at '{csv_path}'. Download HAM10000 from "
                f"the ISIC Archive or Kaggle ('skin-cancer-mnist-ham10000') and place "
                f"it at this path, or use SyntheticLesionDataset for a zero-download demo."
            )

        self.metadata = pd.read_csv(csv_path)
        self.metadata["label_name"] = self.metadata["dx"].map(HAM10000_DX_MAP)
        self.metadata = self.metadata.dropna(subset=["label_name"]).reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                "No rows in the metadata CSV matched a known 'dx' code. "
                "Check that the CSV format matches the standard HAM10000_metadata.csv."
            )

        self.class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
        self.transform = transform or get_eval_transforms()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row["image_id"]
        label = self.class_to_idx[row["label_name"]]

        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at '{img_path}'.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image_rgb)
        return tensor, label


class SyntheticLesionDataset(Dataset):
   

    def __init__(self, num_samples: int = 200, image_size: int = IMAGE_SIZE,
                 transform=None, seed: int = 42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform or get_train_transforms(image_size)
        self.rng = np.random.RandomState(seed)
        self.labels = self.rng.randint(0, len(CLASS_NAMES), size=num_samples)

    def __len__(self):
        return self.num_samples

    def _generate_synthetic_image(self, label_idx: int) -> np.ndarray:
        size = self.image_size
        # Skin-toned background (BGR) with slight texture noise
        base_color = np.array([170, 140, 120], dtype=np.uint8)
        img = np.ones((size, size, 3), dtype=np.uint8) * base_color
        noise = self.rng.randint(-15, 15, (size, size, 3))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

        
        center = (size // 2 + self.rng.randint(-20, 20), size // 2 + self.rng.randint(-20, 20))
        radius = self.rng.randint(30, 70)
        palette = [
            (40, 40, 90),     # 0 Melanoma - dark irregular
            (90, 110, 140),   # 1 Nevus - brownish
            (60, 90, 160),    # 2 BCC - pinkish
            (70, 130, 170),   # 3 AK - reddish scaly
            (100, 100, 100),  # 4 Benign keratosis - grey-brown
            (80, 120, 110),   # 5 Dermatofibroma
            (50, 30, 160),    # 6 Vascular lesion - red
        ]
        lesion_color = palette[label_idx % len(palette)]
        cv2.ellipse(img, center, (radius, int(radius * 0.8)), int(self.rng.randint(0, 180)),
                    0, 360, lesion_color, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        
        for _ in range(self.rng.randint(3, 8)):
            pt1 = (int(self.rng.randint(0, size)), int(self.rng.randint(0, size)))
            pt2 = (pt1[0] + int(self.rng.randint(-80, 80)), pt1[1] + int(self.rng.randint(-80, 80)))
            cv2.line(img, pt1, pt2, (10, 10, 10), thickness=int(self.rng.randint(1, 3)))

        return img

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        image_bgr = self._generate_synthetic_image(label)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb)
        return tensor, label