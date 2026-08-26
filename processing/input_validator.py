from pathlib import Path
from typing import List, Dict, Any


SUPPORTED_RASTER = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def validate_file(path: str) -> Dict[str, Any]:
    """
    Validate a single remote-sensing image input.
    """

    file_path = Path(path)

    result = {
        "valid": False,
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "errors": [],
        "warnings": [],
    }

    if not file_path.exists():
        result["errors"].append("File does not exist.")
        return result

    if file_path.suffix.lower() not in SUPPORTED_RASTER:
        result["errors"].append(
            f"Unsupported format: {file_path.suffix}. "
            f"Supported formats: GeoTIFF/TIFF, PNG, JPEG."
        )
        return result

    if file_path.stat().st_size == 0:
        result["errors"].append("File is empty.")
        return result

    if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        result["warnings"].append(
            "PNG/JPEG accepted for benchmark/demo usage. "
            "GeoTIFF is preferred for geospatial analysis."
        )

    result["valid"] = True
    return result


def validate_inputs(paths: List[str]) -> Dict[str, Any]:
    """
    Validate the number and format of uploaded images.
    """

    result = {
        "valid": False,
        "image_count": len(paths),
        "files": [],
        "errors": [],
        "warnings": [],
    }

    if len(paths) == 0:
        result["errors"].append("No image uploaded.")
        return result

    if len(paths) > 2:
        result["errors"].append(
            "SatQuery AI currently supports a maximum of two images."
        )
        return result

    for path in paths:
        file_result = validate_file(path)
        result["files"].append(file_result)

        if not file_result["valid"]:
            result["errors"].extend(file_result["errors"])

        result["warnings"].extend(file_result["warnings"])

    result["valid"] = len(result["errors"]) == 0

    return result