from pathlib import Path
from typing import Dict, Any

import numpy as np
import rasterio
from PIL import Image


def read_raster_metadata(path: str) -> Dict[str, Any]:
    """
    Read metadata from a remote-sensing raster.
    """

    file_path = Path(path)

    result = {
        "filename": file_path.name,
        "path": str(file_path),
        "format": file_path.suffix.lower(),
        "width": None,
        "height": None,
        "bands": None,
        "dtype": None,
        "crs": None,
        "bounds": None,
        "resolution": None,
        "georeferenced": False,
        "error": None,
    }

    try:
        with rasterio.open(file_path) as src:

            result["width"] = src.width
            result["height"] = src.height
            result["bands"] = src.count
            result["dtype"] = str(src.dtypes[0])

            if src.crs:
                result["crs"] = str(src.crs)
                result["georeferenced"] = True

            result["bounds"] = {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            }

            result["resolution"] = {
                "x": src.res[0],
                "y": src.res[1],
            }

    except Exception as e:
        result["error"] = str(e)

    return result


def load_preview(path: str, max_size: int = 1024) -> Image.Image:
    """
    Load a satellite image and convert it to a displayable image.
    """

    file_path = Path(path)

    with rasterio.open(file_path) as src:

        band_count = min(src.count, 3)

        data = src.read(
            list(range(1, band_count + 1)),
            out_shape=(
                band_count,
                min(src.height, max_size),
                min(src.width, max_size),
            ),
        )

    normalized = []

    for band in data:

        band = band.astype(np.float32)

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high <= low:
            high = low + 1

        band = np.clip(
            (band - low) / (high - low),
            0,
            1
        )

        band = (band * 255).astype(np.uint8)

        normalized.append(band)

    if len(normalized) == 1:
        return Image.fromarray(normalized[0], mode="L")

    if len(normalized) == 2:
        array = np.stack(
            [
                normalized[0],
                normalized[1],
                normalized[1]
            ],
            axis=-1
        )
    else:
        array = np.stack(
            [
                normalized[0],
                normalized[1],
                normalized[2]
            ],
            axis=-1
        )

    return Image.fromarray(array, mode="RGB")