"""
Image preprocessing module.

Implements the DullRazor algorithm for dermoscopic hair-artifact removal
using OpenCV morphological black-hat filtering + inpainting, plus resizing
and tensor normalization utilities used before every model inference call.

Reference: Lee et al., "DullRazor: A software approach to hair removal
from images", Computers in Biology and Medicine, 1997.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def remove_hair_dullrazor(image_bgr: np.ndarray, kernel_size: int = 17,
                           inpaint_radius: int = 1) -> np.ndarray:
    """
    Removes hair artifacts from a dermoscopic image using the DullRazor
    algorithm: grayscale conversion -> morphological black-hat filter ->
    thresholding to build a binary hair mask -> inpainting over the mask.

    Args:
        image_bgr: Input image in BGR format (as read by cv2.imread /
            produced by cv2.cvtColor(..., COLOR_RGB2BGR)).
        kernel_size: Size of the structuring element used for the
            black-hat transform. Larger values catch thicker hair strands.
        inpaint_radius: Radius parameter passed to cv2.inpaint.

    Returns:
        Hair-removed image in BGR format, same shape as the input.

    Raises:
        ValueError: if the input image is empty or None.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or None.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Structuring element: black-hat highlights dark, thin structures
    # (hair strands) against the lighter, smoother skin background.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Threshold the black-hat response to obtain a binary hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Slightly dilate the mask so inpainting fully covers hair edges/fringes
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Inpaint the masked hair regions using Telea's fast marching method
    clean_image = cv2.inpaint(image_bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return clean_image


def resize_image(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resizes an image to size x size using area interpolation (best for downscaling)."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or None.")
    return cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_AREA)


def bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Converts a BGR OpenCV image into a normalized CHW torch tensor
    ready for model inference (ImageNet mean/std normalization).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image_rgb)


def preprocess_pipeline(image_bgr: np.ndarray, remove_hair: bool = True):
    """
    Full preprocessing pipeline: optional hair removal -> resize -> tensor
    conversion. Convenience wrapper used by scripts that don't need each
    intermediate step separately (the Streamlit app calls the pieces
    individually so it can display the raw vs. clean images side by side).

    Returns:
        Tuple of (clean_image_bgr, resized_clean_bgr, input_tensor)
    """
    clean = remove_hair_dullrazor(image_bgr) if remove_hair else image_bgr.copy()
    resized = resize_image(clean, IMAGE_SIZE)
    tensor = bgr_to_tensor(resized)
    return clean, resized, tensor
