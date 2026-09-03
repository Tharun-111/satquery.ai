from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


_FEATURE_PROCESSOR = None
_FEATURE_MODEL = None


def _semantic_change_score(image_before, image_after):
    """Compare images with a lightweight pretrained MobileNetV2 encoder."""
    global _FEATURE_PROCESSOR, _FEATURE_MODEL

    if _FEATURE_PROCESSOR is None or _FEATURE_MODEL is None:
        _FEATURE_PROCESSOR = AutoImageProcessor.from_pretrained(
            "google/mobilenet_v2_1.0_224"
        )
        _FEATURE_MODEL = AutoModel.from_pretrained(
            "google/mobilenet_v2_1.0_224"
        )
        _FEATURE_MODEL.eval()

    before_tensor = torch.from_numpy(image_before).permute(2, 0, 1)
    after_tensor = torch.from_numpy(image_after).permute(2, 0, 1)
    before_image = before_tensor.permute(1, 2, 0).numpy()
    after_image = after_tensor.permute(1, 2, 0).numpy()
    inputs = _FEATURE_PROCESSOR(
        images=[before_image, after_image],
        return_tensors="pt",
    )

    with torch.no_grad():
        features = _FEATURE_MODEL(**inputs).last_hidden_state
        features = features.mean(dim=1)
        features = torch.nn.functional.normalize(features, dim=1)

    similarity = float(torch.sum(features[0] * features[1]))
    return max(0.0, min(1.0, 1.0 - similarity))


def _read_image(path):
    """
    Read PNG/JPEG/TIFF image and convert it to RGB.
    """
    path = Path(path)

    return np.asarray(Image.open(path).convert("RGB"))


def _resize_pair(image1, image2):
    """
    Resize the second image to match the first image.
    """
    height, width = image1.shape[:2]

    if image2.shape[:2] != (height, width):
        image2 = np.asarray(
            Image.fromarray(image2).resize(
                (width, height),
                Image.Resampling.BILINEAR
            )
        )

    return image1, image2


def detect_change(image_before, image_after, threshold=30):
    """
    Detect pixel-level changes between two corresponding images.

    Returns:
        change_mask
        change_percentage
        changed_pixels
        total_pixels
    """

    image_before, image_after = _resize_pair(
        image_before,
        image_after
    )

    gray_before = np.dot(
        image_before[..., :3],
        [0.299, 0.587, 0.114]
    )
    gray_after = np.dot(
        image_after[..., :3],
        [0.299, 0.587, 0.114]
    )
    difference = np.abs(gray_before - gray_after)
    change_mask = np.where(difference >= threshold, 255, 0).astype(
        np.uint8
    )

    changed_pixels = int(
        np.count_nonzero(change_mask)
    )

    total_pixels = int(
        change_mask.size
    )

    change_percentage = (
        changed_pixels / total_pixels * 100
        if total_pixels > 0
        else 0
    )

    return {
        "mask": change_mask,
        "change_percentage": change_percentage,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels
    }


def create_change_overlay(
    image_before,
    image_after,
    change_mask
):
    """
    Create a visual evidence image showing detected changes.
    """

    image_before, image_after = _resize_pair(
        image_before,
        image_after
    )

    # Use the later image as the background
    overlay = image_after.copy()

    # Highlight changed pixels
    changed = change_mask > 0

    # Highlight changes using a bright red overlay
    overlay[changed] = [
        255,
        0,
        0
    ]

    # Blend with original
    return (
        image_after.astype(np.float32) * 0.65
        + overlay.astype(np.float32) * 0.35
    ).clip(0, 255).astype(np.uint8)


def analyze_change(
    before_path,
    after_path,
    threshold=30
):
    """
    Complete bi-temporal change analysis.
    """

    before_path = Path(before_path)
    after_path = Path(after_path)

    if not before_path.exists():
        raise FileNotFoundError(
            f"Before image not found: {before_path}"
        )

    if not after_path.exists():
        raise FileNotFoundError(
            f"After image not found: {after_path}"
        )

    before = _read_image(before_path)
    after = _read_image(after_path)

    before, after = _resize_pair(
        before,
        after
    )

    result = detect_change(
        before,
        after,
        threshold
    )

    overlay = create_change_overlay(
        before,
        after,
        result["mask"]
    )

    result["overlay"] = overlay

    result["before_shape"] = list(
        before.shape
    )

    result["after_shape"] = list(
        after.shape
    )

    result["summary"] = (
        f"Detected approximately "
        f"{result['change_percentage']:.2f}% "
        f"changed pixels between the two observations."
    )
    result["semantic_change_score"] = _semantic_change_score(
        before,
        after
    )
    result["summary"] += (
        f" Pretrained MobileNetV2 semantic change score: "
        f"{result['semantic_change_score']:.3f}."
    )

    return result