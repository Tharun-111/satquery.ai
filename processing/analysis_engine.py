from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_image(path):
    """
    Load PNG/JPEG/TIFF image into a NumPy array.
    """
    path = Path(path)

    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        image = Image.open(path).convert("RGB")
        return image, np.array(image)

    # TIFF fallback
    image = Image.open(path).convert("RGB")
    return image, np.array(image)


def basic_image_analysis(path):
    """
    Generate deterministic visual evidence from an image.
    This is the baseline analysis layer used by SatQuery AI.
    """

    image, array = load_image(path)

    height, width = array.shape[:2]

    # RGB statistics
    mean_rgb = array.mean(axis=(0, 1))
    std_rgb = array.std(axis=(0, 1))

    # Simple colour-based water candidate.
    # This is intentionally a baseline heuristic and will later
    # be replaceable by a specialised remote-sensing model.
    r = array[:, :, 0].astype(float)
    g = array[:, :, 1].astype(float)
    b = array[:, :, 2].astype(float)

    water_mask = (
        (b > r * 1.05)
        & (b > g * 0.95)
        & (b > 60)
    )

    ys, xs = np.where(water_mask)

    evidence_image = image.copy()
    draw = ImageDraw.Draw(evidence_image)

    bbox = None

    if len(xs) > 100:
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())

        bbox = [x1, y1, x2, y2]

        draw.rectangle(
            bbox,
            outline="red",
            width=4
        )

        draw.text(
            (x1 + 5, y1 + 5),
            "Water candidate",
            fill="red"
        )

    return {
        "width": width,
        "height": height,
        "mean_rgb": mean_rgb.round(2).tolist(),
        "std_rgb": std_rgb.round(2).tolist(),
        "water_candidate_pixels": int(water_mask.sum()),
        "water_bbox": bbox,
        "evidence_image": evidence_image,
    }