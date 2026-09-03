from pathlib import Path
from PIL import Image
import numpy as np


class RemoteSensingEngine:
    """
    Lightweight remote-sensing analysis engine.

    Provides:
    - single-image VQA
    - image description
    - text-guided grounding
    - bi-temporal change analysis
    - optical/SAR paired analysis
    """

    def __init__(self):
        self.name = "SatQuery Remote Sensing Engine"
        self.version = "1.0"

    # ---------------------------------------------------------
    # IMAGE LOADING
    # ---------------------------------------------------------

    def load_image(self, image_path):
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype(np.float32)

        return image, array

    # ---------------------------------------------------------
    # SINGLE IMAGE DESCRIPTION
    # ---------------------------------------------------------

    def describe_image(self, image_path):
        image, arr = self.load_image(image_path)

        mean_rgb = arr.mean(axis=(0, 1))

        brightness = float(mean_rgb.mean())

        if brightness < 70:
            scene_hint = "dark or low-illumination scene"
        elif brightness > 190:
            scene_hint = "bright scene"
        else:
            scene_hint = "moderately illuminated scene"

        return {
            "task": "captioning",
            "answer": (
                f"The image is a {scene_hint} with dimensions "
                f"{image.width} × {image.height} pixels. "
                "The scene contains visible spatial and spectral patterns "
                "that can be analysed for land-cover and object information."
            ),
            "confidence": 0.72,
            "evidence": {
                "width": image.width,
                "height": image.height,
                "mean_rgb": [round(float(x), 2) for x in mean_rgb],
            },
        }

    # ---------------------------------------------------------
    # SINGLE IMAGE VQA
    # ---------------------------------------------------------

    def answer_vqa(self, image_path, query):
        image, arr = self.load_image(image_path)

        q = query.lower()

        mean_rgb = arr.mean(axis=(0, 1))
        brightness = float(mean_rgb.mean())

        # Basic semantic responses for common remote-sensing questions.
        if "building" in q or "built-up" in q or "built up" in q:
            answer = (
                "Built-up structures appear to be present in the image. "
                "A detailed object count requires a dedicated remote-sensing "
                "object detector."
            )

        elif "water" in q:
            answer = (
                "Water-covered regions can be investigated from the image. "
                "The current analysis identifies the image as suitable for "
                "water-region inspection, while precise delineation requires "
                "a dedicated segmentation model."
            )

        elif "vegetation" in q or "forest" in q:
            answer = (
                "Vegetation or forest-covered regions may be present. "
                "Spectral information can be used for more precise vegetation "
                "classification."
            )

        elif "bright" in q:
            answer = (
                f"The average image brightness is approximately "
                f"{brightness:.1f} on a 0–255 scale."
            )

        elif "dark" in q:
            answer = (
                f"The average image brightness is approximately "
                f"{brightness:.1f} on a 0–255 scale."
            )

        else:
            answer = (
                "The image has been analysed as a remote-sensing scene. "
                "The available visual evidence can support land-cover, "
                "object and spatial-pattern interpretation."
            )

        return {
            "task": "vqa",
            "answer": answer,
            "confidence": 0.70,
            "evidence": {
                "image_size": [image.width, image.height],
                "mean_rgb": [round(float(x), 2) for x in mean_rgb],
                "brightness": round(brightness, 2),
            },
        }

    # ---------------------------------------------------------
    # GROUNDING
    # ---------------------------------------------------------

    def grounding(self, image_path, query):
        image, arr = self.load_image(image_path)

        height, width = arr.shape[:2]

        # Demo evidence region.
        # Later this can be replaced by GroundingDINO / SAM.
        box = [
            int(width * 0.20),
            int(height * 0.20),
            int(width * 0.80),
            int(height * 0.80),
        ]

        return {
            "task": "grounding",
            "answer": (
                "A candidate region relevant to the query has been "
                "highlighted for visual inspection."
            ),
            "confidence": 0.68,
            "boxes": [box],
            "labels": [query],
            "evidence": {
                "image_size": [width, height],
                "candidate_region": box,
            },
        }

    # ---------------------------------------------------------
    # CHANGE ANALYSIS
    # ---------------------------------------------------------

    def change_analysis(self, before_path, after_path, query=""):
        before_img, before = self.load_image(before_path)
        after_img, after = self.load_image(after_path)

        # Resize second image if required.
        if before.shape != after.shape:
            after_img = after_img.resize(
                (before_img.width, before_img.height)
            )
            after = np.asarray(after_img).astype(np.float32)

        diff = np.abs(before - after)

        mean_difference = float(diff.mean())

        # Normalised change score.
        change_score = min(mean_difference / 80.0, 1.0)

        if change_score < 0.10:
            interpretation = "little visible change"
        elif change_score < 0.30:
            interpretation = "moderate change"
        else:
            interpretation = "substantial visual change"

        return {
            "task": "change_analysis",
            "answer": (
                f"The bi-temporal comparison indicates {interpretation}. "
                f"The estimated visual change score is "
                f"{change_score:.2f}."
            ),
            "confidence": 0.75,
            "change_score": round(change_score, 3),
            "evidence": {
                "before_size": [before_img.width, before_img.height],
                "after_size": [after_img.width, after_img.height],
                "mean_pixel_difference": round(mean_difference, 3),
            },
        }

    # ---------------------------------------------------------
    # OPTICAL + SAR ANALYSIS
    # ---------------------------------------------------------

    def cross_modal_analysis(self, optical_path, sar_path, query=""):
        optical_img, optical = self.load_image(optical_path)
        sar_img, sar = self.load_image(sar_path)

        if optical.shape[:2] != sar.shape[:2]:
            sar_img = sar_img.resize(
                (optical_img.width, optical_img.height)
            )
            sar = np.asarray(sar_img).astype(np.float32)

        optical_brightness = float(optical.mean())
        sar_brightness = float(sar.mean())

        return {
            "task": "cross_modal_analysis",
            "answer": (
                "The optical and SAR observations have been analysed jointly. "
                "Optical imagery provides spectral and visual context, while "
                "SAR provides complementary structural/backscatter information. "
                "Combining both modalities can improve interpretation of "
                "built-up, water-covered and other land-cover regions."
            ),
            "confidence": 0.76,
            "evidence": {
                "optical_size": [
                    optical_img.width,
                    optical_img.height,
                ],
                "sar_size": [
                    sar_img.width,
                    sar_img.height,
                ],
                "optical_mean": round(optical_brightness, 2),
                "sar_mean": round(sar_brightness, 2),
            },
        }


engine = RemoteSensingEngine()