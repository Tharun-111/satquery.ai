from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class DisasterAnalyzer:
    """CLIP zero-shot classifier for disaster-related visual evidence."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.labels = [
            "normal terrain with no visible disaster",
            "flooded or water-covered area",
            "building or infrastructure damage",
            "active wildfire or burned area",
            "landslide or debris flow",
            "storm or cyclone damage",
        ]

    def analyze(self, image_path: str, query: str) -> Dict[str, Any]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(path).convert("RGB")
        prompts = [
            f"An aerial satellite image showing {label}."
            for label in self.labels
        ]
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = self.model(**inputs).logits_per_image[0]
            probabilities = logits.softmax(dim=0)

        best_index = int(probabilities.argmax().item())
        scores = [
            {
                "label": label,
                "score": round(float(probability), 4),
            }
            for label, probability in zip(self.labels, probabilities)
        ]
        scores.sort(key=lambda item: item["score"], reverse=True)

        return {
            "answer": (
                f"The model classified the image most strongly as "
                f"'{self.labels[best_index]}'."
            ),
            "confidence": float(probabilities[best_index]),
            "evidence": [
                f"Model: {self.model_name}",
                f"Query: {query}",
                f"Top classification: {self.labels[best_index]}",
                f"Inference device: {self.device}",
            ],
            "classifications": scores,
            "device": self.device,
        }


def create_disaster_model() -> DisasterAnalyzer:
    return DisasterAnalyzer()
