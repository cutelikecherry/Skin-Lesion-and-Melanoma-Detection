
import pickle
from typing import List, Optional, Tuple
 
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
 
from config import CLASS_NAMES
 
 
@torch.no_grad()
def extract_features(backbone: torch.nn.Module, dataset: Dataset, device: str = "cpu",
                      batch_size: int = 16, num_workers: int = 0,
                      desc: str = "Extracting features") -> Tuple[torch.Tensor, torch.Tensor]:

    backbone.eval()
    backbone.to(device)
 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
 
    all_features, all_labels = [], []
    for images, labels in tqdm(loader, desc=desc, unit="batch"):
        images = images.to(device)
        feats = backbone(images)  
        all_features.append(feats.cpu())
        all_labels.append(labels)
 
    if not all_features:
        raise ValueError("Dataset was empty -- nothing to extract features from.")
 
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels
 
 
def save_feature_cache(path: str, train_features: torch.Tensor, train_labels: torch.Tensor,
                        val_features: torch.Tensor, val_labels: torch.Tensor,
                        backbone_name: str, class_names: Optional[List[str]] = None):
    
    payload = {
        "format": "feature_cache_v1",
        "backbone_name": backbone_name,
        "class_names": class_names or CLASS_NAMES,
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
 
 
def load_feature_cache(path: str) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
 
    required_keys = {"backbone_name", "class_names", "train_features", "train_labels",
                      "val_features", "val_labels"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Feature cache at '{path}' is missing keys: {missing}. "
                          f"Was this file really written by extract_features.py?")
    return payload