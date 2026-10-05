
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = None
        self._backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None,
                 device: Optional[str] = None) -> Tuple[np.ndarray, int]:
       
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)

        batch = input_tensor.unsqueeze(0).to(device)
        batch.requires_grad_(True)

        logits = self.model(batch)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire -- verify that target_layer is actually "
                "part of the forward pass executed by model(batch)."
            )

        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)         # (1, C, 1, 1)
        weighted_activations = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,H,W)
        cam = F.relu(weighted_activations).squeeze().cpu().numpy()

        cam = cam - cam.min()
        if cam.max() > 1e-8:
            cam = cam / cam.max()

        return cam, class_idx


def overlay_heatmap_on_image(heatmap: np.ndarray, original_bgr: np.ndarray,
                              alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    h, w = original_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    overlay = cv2.addWeighted(colored_heatmap, alpha, original_bgr, 1 - alpha, 0)
    return overlay