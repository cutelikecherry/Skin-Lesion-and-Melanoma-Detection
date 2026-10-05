
import torch

CLASS_NAMES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion",
]

NUM_CLASSES = len(CLASS_NAMES)

RISK_MAP = {
    "Melanoma": "High",
    "Basal Cell Carcinoma": "High",
    "Actinic Keratosis": "Medium",
    "Melanocytic Nevus": "Low",
    "Benign Keratosis": "Low",
    "Dermatofibroma": "Low",
    "Vascular Lesion": "Low",
}

RISK_COLORS = {
    "High": "#D32F2F",    # red
    "Medium": "#F9A825",  # amber
    "Low": "#2E7D32",     # green
}

IMAGE_SIZE = 300

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
