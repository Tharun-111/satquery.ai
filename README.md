# SatQuery AI

SatQuery AI is a Streamlit assistant for remote-sensing image analysis. It
supports four workflows:

1. **Single Image** - BLIP visual question answering and image captioning.
2. **Object Detection** - Grounding DINO zero-shot detection for buildings,
   roads, water, vehicles, and other queried objects.
3. **Disaster Analysis** - CLIP zero-shot classification for floods, fires,
   infrastructure damage, landslides, storms, and normal terrain.
4. **Bi-Temporal Change Analysis** - pixel-level change detection with a
   lightweight pretrained MobileNetV2 semantic comparison and visual overlay.

## Setup

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m streamlit run app/main.py
```

The first request for a model downloads its weights from Hugging Face. CPU
inference is supported, but a CUDA-enabled PyTorch installation is recommended
for faster inference.

## Input formats

PNG, JPEG, and GeoTIFF files are supported. Change analysis requires a
co-registered BEFORE and AFTER image. Uploaded files are written to
`data/demo/`, and generated change evidence is written to `outputs/`.

## Project layout

- `app/` - Streamlit interface
- `agent/` - routing and specialist coordination
- `models/` - pretrained model integrations
- `processing/` - raster, validation, and change-processing utilities
- `evidence/` - evidence-generation helpers
- `data/samples/` - sample input data

## Notes

The disaster classifier is a general-purpose CLIP zero-shot model, not a
certified emergency-response system. Results should be reviewed by a qualified
analyst before operational use.
