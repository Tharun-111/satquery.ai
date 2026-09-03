import os
from typing import Any, Dict

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class RemoteSensingCaptioner:
    """BLIP image-captioning specialist."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

    def describe(self, image_path: str, query: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
            )

        caption = self.processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        return {
            "answer": caption or "No caption generated.",
            "confidence": 1.0 if caption else 0.0,
            "evidence": [
                f"Model: {self.model_name}",
                f"Query: {query}",
                f"Inference device: {self.device}",
            ],
            "device": self.device,
        }


def create_caption_model() -> RemoteSensingCaptioner:
    return RemoteSensingCaptioner()