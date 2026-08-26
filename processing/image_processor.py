from pathlib import Path

import numpy as np
from PIL import Image
import rasterio


def load_image(file_path):
    """
    Load a PNG/JPEG/TIFF/GeoTIFF image.

    Returns:
        image_array: numpy array in H x W x C format
        metadata: dictionary containing image information
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    extension = path.suffix.lower()

    # ---------------------------------------------------------
    # PNG / JPEG
    # ---------------------------------------------------------
    if extension in {".png", ".jpg", ".jpeg"}:

        image = Image.open(path).convert("RGB")
        image_array = np.array(image)

        metadata = {
            "filename": path.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "bands": 3,
            "georeferenced": False,
            "crs": None,
        }

        return image_array, metadata

    # ---------------------------------------------------------
    # TIFF / GeoTIFF
    # ---------------------------------------------------------
    if extension in {".tif", ".tiff"}:

        with rasterio.open(path) as src:

            data = src.read()

            # Rasterio gives:
            # Bands x Height x Width

            # Convert to:
            # Height x Width x Bands

            image_array = np.transpose(data, (1, 2, 0))

            metadata = {
                "filename": path.name,
                "format": "GeoTIFF/TIFF",
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "georeferenced": src.crs is not None,
                "crs": str(src.crs) if src.crs else None,
                "transform": str(src.transform),
            }

        return image_array, metadata

    raise ValueError(
        f"Unsupported image format: {extension}"
    )


def normalize_band(band):
    """
    Normalize a single image band to uint8.
    """

    band = band.astype(np.float32)

    minimum = np.nanpercentile(band, 2)
    maximum = np.nanpercentile(band, 98)

    if maximum <= minimum:
        return np.zeros_like(band, dtype=np.uint8)

    normalized = (band - minimum) / (maximum - minimum)

    normalized = np.clip(normalized, 0, 1)

    return (normalized * 255).astype(np.uint8)


def create_display_image(image_array):
    """
    Convert arbitrary-band satellite imagery
    into an RGB image suitable for Streamlit.
    """

    bands = image_array.shape[2]

    # RGB image
    if bands >= 3:

        red = normalize_band(image_array[:, :, 0])
        green = normalize_band(image_array[:, :, 1])
        blue = normalize_band(image_array[:, :, 2])

        rgb = np.stack(
            [red, green, blue],
            axis=2
        )

        return rgb

    # Single-band image
    if bands == 1:

        band = normalize_band(image_array[:, :, 0])

        return np.stack(
            [band, band, band],
            axis=2
        )

    raise ValueError(
        f"Unsupported number of bands: {bands}"
    )


def calculate_statistics(image_array):
    """
    Calculate simple image statistics.

    These statistics become part of the evidence
    returned by the processing tool.
    """

    statistics = {}

    for band_index in range(image_array.shape[2]):

        band = image_array[:, :, band_index].astype(
            np.float32
        )

        statistics[f"band_{band_index + 1}"] = {
            "minimum": float(np.nanmin(band)),
            "maximum": float(np.nanmax(band)),
            "mean": float(np.nanmean(band)),
            "std": float(np.nanstd(band)),
        }

    return statistics


def analyze_image(file_path):
    """
    Main image-processing entry point.

    Returns a complete evidence package.
    """

    image_array, metadata = load_image(file_path)

    display_image = create_display_image(
        image_array
    )

    statistics = calculate_statistics(
        image_array
    )

    evidence = {
        "metadata": metadata,
        "statistics": statistics,
        "shape": list(image_array.shape),
    }

    return {
        "image_array": image_array,
        "display_image": display_image,
        "metadata": metadata,
        "statistics": statistics,
        "evidence": evidence,
    }