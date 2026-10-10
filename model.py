import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torchvision import models

from config import CLASS_NAMES, NUM_CLASSES, RISK_MAP


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    probabilities: List[float]
    risk_level: str
    class_names: List[str]


class SkinLesionClassifier(nn.Module):

    def __init__(self, backbone_name: str = "efficientnet_b4", num_classes: int = NUM_CLASSES,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.efficientnet_b4(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            # Last conv block before global pooling -- used as the Grad-CAM target layer
            self.target_layer_name = "backbone.features.8.0"
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.target_layer_name = "backbone.layer4.2.conv3"
        else:
            raise ValueError(f"Unsupported backbone: '{backbone_name}'. "
                              f"Choose 'efficientnet_b4' or 'resnet50'.")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier_head(features)
        return logits

    def get_target_layer(self) -> nn.Module:
        """Resolves and returns the nn.Module used as the Grad-CAM target (last conv layer)."""
        module = self
        for attr in self.target_layer_name.split("."):
            module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
        return module


def build_model(backbone_name: str = "efficientnet_b4", pretrained: bool = True,
                 device: Optional[str] = None) -> SkinLesionClassifier:
    """Factory function that builds the model and moves it to the correct device."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SkinLesionClassifier(backbone_name=backbone_name, pretrained=pretrained)
    model.to(device)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str,
                     device: Optional[str] = None) -> nn.Module:
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def save_lightweight_checkpoint(model: "SkinLesionClassifier", path: str):
   
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "format": "lightweight_head_v1",
        "backbone_name": model.backbone_name,
        "class_names": CLASS_NAMES,
        "classifier_head_state_dict": model.classifier_head.state_dict(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_lightweight_checkpoint(path: str, device: Optional[str] = None) -> "SkinLesionClassifier":
   
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at '{path}'.")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or payload.get("format") != "lightweight_head_v1":
        raise ValueError(
            f"'{path}' does not look like a lightweight checkpoint written by "
            f"save_lightweight_checkpoint(). Use load_checkpoint() for full .pth "
            f"state-dict files instead."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone_name=payload["backbone_name"], pretrained=True, device=device)
    model.classifier_head.load_state_dict(payload["classifier_head_state_dict"])
    model.to(device)
    model.eval()
    return model


def compute_risk_level(predicted_class: str, confidence: float) -> str:
   
    base_risk = RISK_MAP.get(predicted_class, "Low")
    if base_risk == "High":
        return "High"
    if confidence < 0.5:
        return "Medium"
    return base_risk


@torch.no_grad()
def predict(model: nn.Module, input_tensor: torch.Tensor,
            device: Optional[str] = None) -> PredictionResult:
    """
    Runs inference on a single preprocessed image tensor (C, H, W) and
    returns the predicted class, per-class probabilities, and risk level.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    batch = input_tensor.unsqueeze(0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().tolist()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    risk_level = compute_risk_level(predicted_class, confidence)

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probs,
        risk_level=risk_level,
        class_names=CLASS_NAMES,
    )
