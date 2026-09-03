# SatQuery AI

> An AI-assisted remote-sensing application for image understanding,
> object localization, disaster screening, and bi-temporal change analysis.

SatQuery AI is an interactive Streamlit application that combines geospatial
image validation, deterministic task routing, pretrained vision models, and
auditable evidence. It is designed for rapid exploration of optical,
multispectral, SAR, and other satellite imagery.

[**Launch the live application**](https://satqueryai-thxxxn.streamlit.app/)

## Features

### 1. Single Image Analysis

Upload one image and ask a natural-language question. The application routes
description requests to BLIP image captioning and visual questions to BLIP
Visual Question Answering.

Example questions:

- "Describe the scene."
- "What land-cover types are visible?"
- "Is there water in the image?"
- "How bright is the image?"

### 2. Object Detection

Grounding DINO performs text-guided, zero-shot object detection. The query is
converted into candidate labels and the model returns bounding regions,
confidence scores, and an object count.

Example queries:

- "Find buildings."
- "Locate roads and vehicles."
- "Highlight the water body."

Object counts are model detections, not a guaranteed cadastral or survey-grade
count. Detection quality depends on image resolution, scene content, and model
confidence thresholds.

### 3. Disaster Analysis

CLIP performs zero-shot classification against disaster-related visual
categories:

- Normal terrain
- Flooding or water coverage
- Building or infrastructure damage
- Wildfire or burned area
- Landslide or debris flow
- Storm or cyclone damage

This provides a rapid screening signal without requiring a task-specific
training dataset. It is not a certified emergency-response or damage-assessment
system.

### 4. Bi-Temporal Change Analysis

Upload a BEFORE and AFTER image pair to compare two observations. The workflow:

1. Reads and normalizes both images.
2. Resizes the AFTER image when dimensions differ.
3. Detects pixel-level differences with OpenCV.
4. Compares pretrained MobileNetV2 image features for a semantic change score.
5. Produces a red change overlay and downloadable PNG evidence.

The two images should be spatially aligned and captured using comparable
conditions for meaningful results.

## Model stack

| Workflow | Model or method | Purpose |
| --- | --- | --- |
| Image captioning | `Salesforce/blip-image-captioning-base` | Generate scene descriptions |
| Visual question answering | `Salesforce/blip-vqa-base` | Answer image questions |
| Object detection | `IDEA-Research/grounding-dino-tiny` | Text-guided region detection |
| Disaster screening | `openai/clip-vit-base-patch32` | Zero-shot disaster classification |
| Change analysis | OpenCV + `google/mobilenet_v2_1.0_224` | Pixel and semantic comparison |

Model weights are downloaded from Hugging Face on first use and cached locally.
The application runs on CPU, although a CUDA-enabled PyTorch installation is
recommended for faster inference.

## System architecture

```text
Streamlit UI
    |
    v
Image validation and raster preview
    |
    v
Deterministic query router
    |
    +--> BLIP VQA / captioning
    +--> Grounding DINO object detection
    +--> CLIP disaster classification
    +--> OpenCV + MobileNetV2 change analysis
    |
    v
Results, confidence, evidence, and downloadable outputs
```

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- Internet access on first model use
- At least 4 GB RAM recommended for CPU inference
- GPU optional; CUDA can substantially improve inference speed

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Running the application

From the repository root:

```powershell
python -m streamlit run app/main.py
```

Streamlit prints a local URL, normally:

```text
http://localhost:8501
```

To select a different port:

```powershell
python -m streamlit run app/main.py --server.port 8502
```

## Using the application

### Single-image workflows

1. Select **Single Image**, **Object Detection**, or **Disaster Analysis**.
2. Upload a PNG, JPEG, or GeoTIFF image.
3. Wait for validation to complete.
4. Enter a focused natural-language query.
5. Select **Analyze Image**.

### Change workflow

1. Select **Bi-Temporal Change Analysis**.
2. Upload the BEFORE observation.
3. Upload the AFTER observation.
4. Enter a question describing the comparison.
5. Select **Analyze Change**.
6. Review the change percentage, semantic score, overlay, and evidence file.

Generated evidence is saved as:

```text
outputs/change_evidence.png
```

## Supported input formats

- PNG
- JPG / JPEG
- TIFF / GeoTIFF

GeoTIFF files are converted to displayable previews through the raster
processing utilities. Preserve geospatial alignment when preparing a
bi-temporal pair.

## Repository structure

```text
.
├── app/
│   └── main.py                    # Streamlit application
├── agent/
│   ├── router.py                  # Deterministic query routing
│   ├── controller.py              # Execution planning
│   ├── executor.py                # Specialist execution
│   └── validator.py               # Input validation
├── models/
│   ├── caption_model.py           # BLIP captioning
│   ├── disaster_model.py          # CLIP disaster screening
│   ├── grounding/                 # Grounding DINO integration
│   └── vqa/                       # BLIP VQA integration
├── processing/
│   ├── raster_processor.py        # Raster loading and previews
│   ├── image_processor.py         # Image analysis utilities
│   └── change_detectoe.py         # Bi-temporal change analysis
├── evidence/                      # Evidence-generation helpers
├── data/samples/                  # Small sample data
├── requirements.txt               # Python dependencies
└── test_validator.py              # Existing validation test
```

## Validation and testing

Run the existing validator test:

```powershell
python test_validator.py
```

Run a syntax check for the application and model modules:

```powershell
python -m py_compile app/main.py models/*.py processing/*.py
```

## Troubleshooting

### Models are downloading slowly

The first inference downloads model weights. Keep the terminal open and allow
the download to finish. Later runs use the local Hugging Face cache.

### CPU inference is slow

This is expected for BLIP and Grounding DINO on CPU. Use a CUDA-enabled PyTorch
installation and compatible NVIDIA drivers when available.

### No objects are detected

Use a specific query such as "buildings", "roads", or "water", ensure the image
has sufficient resolution, and avoid very low-confidence scenes.

### Change analysis reports unexpected differences

Confirm that the BEFORE and AFTER images are co-registered, have comparable
lighting and resolution, and represent the same geographic area.

### Port already in use

Start Streamlit on another port:

```powershell
python -m streamlit run app/main.py --server.port 8502
```

## Responsible use

SatQuery AI is a research and demonstration application. Model outputs can be
incorrect, incomplete, or sensitive to image quality and prompt wording.
Disaster classifications and object counts must be reviewed by a qualified
analyst before being used for operational, legal, safety, or emergency
decisions.

Do not upload confidential imagery unless the deployment environment and model
hosting policy have been reviewed. Hugging Face model downloads and inference
may involve local caches and external network access.

## License and attribution

This repository integrates open-source libraries and pretrained models. Review
the license and usage terms of each dependency and model before redistribution
or commercial deployment.
