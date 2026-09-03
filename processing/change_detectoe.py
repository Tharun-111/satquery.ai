from pathlib import Path

import cv2
import numpy as np
import torch
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

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    # Handle grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Handle BGR / BGRA
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

    return image


def _resize_pair(image1, image2):
    """
    Resize the second image to match the first image.
    """
    height, width = image1.shape[:2]

    if image2.shape[:2] != (height, width):
        image2 = cv2.resize(
            image2,
            (width, height),
            interpolation=cv2.INTER_LINEAR
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

    # Convert to grayscale
    gray_before = cv2.cvtColor(
        image_before,
        cv2.COLOR_RGB2GRAY
    )

    gray_after = cv2.cvtColor(
        image_after,
        cv2.COLOR_RGB2GRAY
    )

    # Absolute pixel difference
    difference = cv2.absdiff(
        gray_before,
        gray_after
    )

    # Threshold
    _, change_mask = cv2.threshold(
        difference,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # Remove small noise
    kernel = np.ones((3, 3), np.uint8)

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel
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
    result = cv2.addWeighted(
        image_after,
        0.65,
        overlay,
        0.35,
        0
    )

    return result


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