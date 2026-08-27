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
    ):
        self.model_name = model_name

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
        Locate regions relevant to the supplied query.
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
                "error": "Image file does not exist.",
            }

        try:
            # IMPORTANT:
            # GeoTIFF files cannot always be opened directly
            # with PIL. Use the existing SatQuery raster
            # preview pipeline to convert the raster to RGB.
            image = load_preview(
                str(path)
            )

            labels = self._build_labels(
                query
            )

            detections = self.detector(
                image,
                candidate_labels=labels,
            )

            regions: List[Dict[str, Any]] = []

            for detection in detections:

                box = detection.get(
                    "box",
                    {}
                )

                regions.append(
                    {
                        "label": detection.get(
                            "label",
                            ""
                        ),
                        "score": float(
                            detection.get(
                                "score",
                                0.0
                            )
                        ),
                        "box": {
                            "xmin": int(
                                box.get(
                                    "xmin",
                                    0
                                )
                            ),
                            "ymin": int(
                                box.get(
                                    "ymin",
                                    0
                                )
                            ),
                            "xmax": int(
                                box.get(
                                    "xmax",
                                    0
                                )
                            ),
                            "ymax": int(
                                box.get(
                                    "ymax",
                                    0
                                )
                            ),
                        },
                    }
                )

            if regions:

                best = max(
                    regions,
                    key=lambda item:
                    item["score"],
                )

                answer = (
                    f"Located "
                    f"{best['label']} "
                    f"with confidence "
                    f"{best['score']:.2f}."
                )

                confidence = best["score"]

            else:

                answer = (
                    "No matching region "
                    "was detected."
                )

                confidence = 0.0

            evidence = [
                f"Source image: {path.name}",
                f"Grounding query: {query}",
                f"Detected regions: {len(regions)}",
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