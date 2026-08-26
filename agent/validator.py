from pathlib import Path

import rasterio


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def validate_image(file_path):
    """
    Validate a remote-sensing image and return metadata.
    """

    path = Path(file_path)

    result = {
        "valid": False,
        "filename": path.name,
        "extension": path.suffix.lower(),
        "format": None,
        "width": None,
        "height": None,
        "bands": None,
        "crs": None,
        "georeferenced": False,
        "message": "",
    }

    if not path.exists():
        result["message"] = "File does not exist."
        return result

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        result["message"] = (
            f"Unsupported file type: {path.suffix}. "
            "Supported formats: TIFF, GeoTIFF, PNG and JPEG."
        )
        return result

    # Rasterio handles TIFF/GeoTIFF.
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            with rasterio.open(path) as src:
                result["valid"] = True
                result["format"] = "GeoTIFF/TIFF"
                result["width"] = src.width
                result["height"] = src.height
                result["bands"] = src.count

                if src.crs:
                    result["crs"] = str(src.crs)
                    result["georeferenced"] = True

                result["message"] = "Image validated successfully."

        except Exception as exc:
            result["message"] = f"Could not read raster: {exc}"

        return result

    # PNG/JPEG are allowed for benchmark/demo imagery.
    try:
        from PIL import Image

        with Image.open(path) as image:
            result["valid"] = True
            result["format"] = image.format
            result["width"] = image.width
            result["height"] = image.height
            result["bands"] = len(image.getbands())
            result["message"] = "Benchmark image validated successfully."

    except Exception as exc:
        result["message"] = f"Could not read image: {exc}"

    return result