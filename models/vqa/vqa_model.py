"""
SatQuery AI - Remote Sensing VQA Model

Connects a satellite image to the BLIP Visual Question Answering model.

Pipeline:
GeoTIFF / JPG / PNG
        ↓
Raster preview conversion
        ↓
RGB image
        ↓
BLIP VQA
        ↓
Answer
"""

import os
from typing import Dict, Any

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


class RemoteSensingVQA:
    """BLIP-based Visual Question Answering specialist."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
    ):
        self.name = "Remote-Sensing VQA Tool"
        self.task = "vqa"
        self.model_name = model_name

        # Use CUDA when available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading VQA model: {self.model_name}")
        print(f"Using device: {self.device}")

        # Load processor
        self.processor = BlipProcessor.from_pretrained(
            self.model_name
        )

        # Load model
        self.model = BlipForQuestionAnswering.from_pretrained(
            self.model_name
        )

        # Move model to GPU/CPU
        self.model.to(self.device)

        # Evaluation mode
        self.model.eval()

        print("VQA model loaded successfully.")

    def _load_image(self, image_path: str) -> Image.Image:
        """
        Load an image.

        Normal images are opened directly.

        GeoTIFF satellite images are converted through
        SatQuery's raster processor.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        extension = os.path.splitext(image_path)[1].lower()

        # GeoTIFF / TIFF
        if extension in [".tif", ".tiff"]:
            from processing.raster_processor import load_preview

            image = load_preview(image_path)

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            return image.convert("RGB")

        # JPG / PNG / other normal images
        image = Image.open(image_path)

        return image.convert("RGB")

    def answer(
        self,
        image_path: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Answer a question about a satellite image.
        """

        try:
            # Validate question
            if not question or not question.strip():
                return {
                    "answer": "No question provided.",
                    "confidence": 0.0,
                    "evidence": [],
                    "image_path": image_path,
                    "question": question,
                    "device": self.device,
                }

            # Load image
            image = self._load_image(image_path)

            # Prepare BLIP inputs
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt",
            )

            # Move tensors to GPU/CPU
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # Run inference
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=30,
                )

            # Decode answer
            answer = self.processor.decode(
                output[0],
                skip_special_tokens=True,
            ).strip()

            if not answer:
                answer = "No answer generated."

            return {
                "answer": answer,
                "confidence": 1.0,
                "evidence": [],
                "image_path": image_path,
                "question": question,
                "device": self.device,
            }

        except Exception as e:
            return {
                "answer": "VQA inference failed.",
                "confidence": 0.0,
                "evidence": [],
                "image_path": image_path,
                "question": question,
                "device": self.device,
                "error": str(e),
            }


def create_vqa_model() -> RemoteSensingVQA:
    """
    Factory function used by the SatQuery AI agent.
    """

    return RemoteSensingVQA()