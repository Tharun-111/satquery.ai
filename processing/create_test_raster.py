import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path


output_path = Path("data/samples/test_satellite.tif")

output_path.parent.mkdir(parents=True, exist_ok=True)

width = 512
height = 512
bands = 4

data = np.random.randint(
    0,
    10000,
    size=(bands, height, width),
    dtype=np.uint16
)

transform = from_origin(
    80.0,       # longitude
    13.0,       # latitude
    0.0001,     # pixel width
    0.0001      # pixel height
)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=height,
    width=width,
    count=bands,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:

    dst.write(data)

print(f"Created test raster: {output_path}")