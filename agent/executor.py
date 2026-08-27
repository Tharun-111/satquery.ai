"""
SatQuery AI - Specialist Executor

Provides a common execution interface for all
remote-sensing specialist tools.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from evidence.evidence_generator import generate_evidence


@dataclass
class ToolResult:
    task: str
    specialist: str
    answer: str
    confidence: float
    evidence: List[str]
    outputs: Dict[str, Any]


def execute_specialist(
    task: str,
    specialist: str,
    images=None,
    query: str = "",
    metadata=None,
) -> Dict[str, Any]:
    """
    Execute a specialist workflow.
    """

    if images is None:
        images = []

    if metadata is None:
        metadata = {}

    # ---------------------------------------------------------
    # VQA specialist
    # ---------------------------------------------------------

    if task == "vqa":

        if len(images) < 1:
            return asdict(
                ToolResult(
                    task=task,
                    specialist=specialist,
                    answer="No image supplied.",
                    confidence=0.0,
                    evidence=[],
                    outputs={
                        "image_count": 0,
                        "query": query,
                        "metadata": metadata,
                    },
                )
            )

        image_path = images[0]

        try:
            from models.vqa.vqa_model import create_vqa_model

            model = create_vqa_model()

            result = model.answer(
                image_path=image_path,
                question=query,
            )

            answer = result.get("answer", "")
            confidence = float(result.get("confidence", 0.0))
            device = result.get("device", "unknown")

            evidence = generate_evidence(
                image_path=image_path,
                query=query,
                answer=answer,
                confidence=confidence,
                device=device,
            )

            return asdict(
                ToolResult(
                    task=task,
                    specialist=specialist,
                    answer=answer,
                    confidence=confidence,
                    evidence=evidence,
                    outputs={
                        "image_count": len(images),
                        "query": query,
                        "metadata": metadata,
                        "image_path": image_path,
                        "device": device,
                    },
                )
            )

        except Exception as exc:

            return asdict(
                ToolResult(
                    task=task,
                    specialist=specialist,
                    answer="VQA inference failed.",
                    confidence=0.0,
                    evidence=[],
                    outputs={
                        "image_count": len(images),
                        "query": query,
                        "metadata": metadata,
                        "error": str(exc),
                    },
                )
            )

    # ---------------------------------------------------------
    # Other specialists
    # ---------------------------------------------------------

    return asdict(
        ToolResult(
            task=task,
            specialist=specialist,
            answer="Specialist execution not connected yet.",
            confidence=0.0,
            evidence=[],
            outputs={
                "image_count": len(images),
                "query": query,
                "metadata": metadata,
            },
        )
    )