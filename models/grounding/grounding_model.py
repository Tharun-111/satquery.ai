"""
SatQuery AI - Remote-Sensing Grounding Model

Uses Grounding DINO to locate regions described by
a natural-language query in remote-sensing imagery.
"""

from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import pipeline

from processing.raster_processor import load_preview


class RemoteSensingGrounding:
    """
    Remote-sensing spatial grounding specialist.
    """

    name = "Remote-Sensing Grounding Tool"

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        threshold: float = 0.30,
    ):
        self.model_name = model_name
        self.threshold = threshold

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading Grounding model: {model_name}"
        )
        print(
            f"Using device: {self.device}"
        )
        print(
            f"Detection threshold: {self.threshold}"
        )

        device_index = (
            0
            if self.device == "cuda"
            else -1
        )

        self.detector = pipeline(
            "zero-shot-object-detection",
            model=model_name,
            device=device_index,
        )

        print(
            "Grounding model loaded successfully."
        )

    def ground(
        self,
        image_path: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Locate reliable regions relevant to the query.
        """

        path = Path(image_path)

        if not path.exists():
            return {
                "answer": "Grounding failed.",
                "confidence": 0.0,
                "evidence": [],
                "image_path": image_path,
                "query": query,
                "device": self.device,
                "regions": [],
                "error": "Image file does not exist.",
            }

        try:

            # -------------------------------------------------
            # Convert GeoTIFF to displayable RGB image
            # -------------------------------------------------

            image = load_preview(
                str(path)
            )

            image_width, image_height = image.size

            # -------------------------------------------------
            # Build candidate labels
            # -------------------------------------------------

            labels = self._build_labels(
                query
            )

            # -------------------------------------------------
            # Grounding inference
            # -------------------------------------------------

            detections = self.detector(
                image,
                candidate_labels=labels,
                threshold=self.threshold,
            )

            regions: List[Dict[str, Any]] = []

            # -------------------------------------------------
            # Validate detections
            # -------------------------------------------------

            for detection in detections:

                box = detection.get(
                    "box",
                    {}
                )

                xmin = int(
                    box.get("xmin", 0)
                )
                ymin = int(
                    box.get("ymin", 0)
                )
                xmax = int(
                    box.get("xmax", 0)
                )
                ymax = int(
                    box.get("ymax", 0)
                )

                score = float(
                    detection.get(
                        "score",
                        0.0
                    )
                )

                label = detection.get(
                    "label",
                    ""
                )

                # ---------------------------------------------
                # Reject invalid boxes
                # ---------------------------------------------

                if xmax <= xmin or ymax <= ymin:
                    continue

                box_width = xmax - xmin
                box_height = ymax - ymin

                box_area = (
                    box_width *
                    box_height
                )

                image_area = (
                    image_width *
                    image_height
                )

                area_ratio = (
                    box_area /
                    image_area
                )

                # ---------------------------------------------
                # Reject boxes covering almost the entire image
                #
                # These were observed with the current model
                # on test_satellite.tif.
                # ---------------------------------------------

                if area_ratio >= 0.95:
                    continue

                regions.append(
                    {
                        "label": label,
                        "score": score,
                        "box": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                        },
                        "area_ratio": round(
                            area_ratio,
                            4
                        ),
                    }
                )

            # -------------------------------------------------
            # Sort strongest detections first
            # -------------------------------------------------

            regions.sort(
                key=lambda item:
                item["score"],
                reverse=True,
            )

            # -------------------------------------------------
            # Build answer
            # -------------------------------------------------

            if regions:

                best = regions[0]

                answer = (
                    f"Located "
                    f"{best['label']} "
                    f"with confidence "
                    f"{best['score']:.2f}."
                )

                confidence = best["score"]

            else:

                answer = (
                    "No reliable matching "
                    "region was detected."
                )

                confidence = 0.0

            # -------------------------------------------------
            # Evidence
            # -------------------------------------------------

            evidence = [
                f"Source image: {path.name}",
                f"Grounding query: {query}",
                f"Candidate labels: {labels}",
                f"Detection threshold: {self.threshold:.2f}",
                f"Reliable regions: {len(regions)}",
                f"Inference device: {self.device}",
            ]

            return {
                "answer": answer,
                "confidence": confidence,
                "evidence": evidence,
                "image_path": image_path,
                "query": query,
                "device": self.device,
                "regions": regions,
            }

        except Exception as error:

            return {
                "answer":
                    "Grounding inference failed.",
                "confidence": 0.0,
                "evidence": [],
                "image_path": image_path,
                "query": query,
                "device": self.device,
                "regions": [],
                "error": str(error),
            }

    @staticmethod
    def _build_labels(
        query: str,
    ) -> List[str]:
        """
        Convert a natural-language query into
        candidate grounding labels.
        """

        query_lower = query.lower()

        known_labels = [
            "water",
            "river",
            "lake",
            "ocean",
            "building",
            "buildings",
            "road",
            "roads",
            "vehicle",
            "vehicles",
            "airplane",
            "ship",
            "forest",
            "tree",
            "trees",
            "field",
            "farmland",
            "urban area",
        ]

        matched = [
            label
            for label in known_labels
            if label in query_lower
        ]

        if matched:
            return matched

        return [query]


def create_grounding_model() -> RemoteSensingGrounding:
    """
    Factory function used by SatQuery AI.
    """

    return RemoteSensingGrounding()