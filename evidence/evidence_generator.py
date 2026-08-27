"""
SatQuery AI - Evidence Generator

Creates simple, auditable evidence records for specialist
model results.

The first version focuses on:
- input image information
- model answer
- confidence
- question/query
- processing device

Later versions can add:
- bounding boxes
- segmentation masks
- change maps
- geographic coordinates
- spectral evidence
"""


from pathlib import Path
from typing import Any, Dict, List


def generate_evidence(
    image_path: str,
    query: str,
    answer: str,
    confidence: float,
    device: str = "unknown",
) -> List[str]:
    """
    Generate human-readable evidence statements.

    Parameters
    ----------
    image_path:
        Path to the image used for inference.

    query:
        User's question/query.

    answer:
        Model-generated answer.

    confidence:
        Model confidence score.

    device:
        Device used for inference.

    Returns
    -------
    List[str]
        Evidence statements.
    """

    evidence = []

    path = Path(image_path)

    if path.exists():
        evidence.append(
            f"Source image: {path.name}"
        )
    else:
        evidence.append(
            f"Source image: {image_path}"
        )

    evidence.append(
        f"Question: {query}"
    )

    evidence.append(
        f"Model answer: {answer}"
    )

    evidence.append(
        f"Model confidence: {confidence:.2f}"
    )

    evidence.append(
        f"Inference device: {device}"
    )

    return evidence


def build_evidence_record(
    image_path: str,
    query: str,
    answer: str,
    confidence: float,
    device: str = "unknown",
) -> Dict[str, Any]:
    """
    Build a structured evidence record.
    """

    return {
        "image_path": image_path,
        "query": query,
        "answer": answer,
        "confidence": confidence,
        "device": device,
        "evidence": generate_evidence(
            image_path=image_path,
            query=query,
            answer=answer,
            confidence=confidence,
            device=device,
        ),
    }