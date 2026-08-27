"""
SatQuery AI - Specialist Executor

Provides a common execution interface for all
remote-sensing specialist tools.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


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

    Currently connected:
        - VQA

    Other specialists will be connected later.
    """

    if images is None:
        images = []

    if metadata is None:
        metadata = {}

    # ---------------------------------------------------------
    # VQA
    # ---------------------------------------------------------

    if task == "vqa":

        if len(images) == 0:
            result = ToolResult(
                task=task,
                specialist=specialist,
                answer="No image was provided.",
                confidence=0.0,
                evidence=[],
                outputs={
                    "image_count": 0,
                    "query": query,
                    "metadata": metadata,
                },
            )

            return asdict(result)

        try:
            # Import only when VQA is actually requested.
            from models.vqa.vqa_model import create_vqa_model

            # Create/load the VQA model.
            model = create_vqa_model()

            # Use the first image for VQA.
            image_path = images[0]

            # Run real VQA inference.
            vqa_result = model.answer(
                image_path=image_path,
                question=query,
            )

            result = ToolResult(
                task=task,
                specialist=specialist,
                answer=vqa_result.get(
                    "answer",
                    "No answer generated.",
                ),
                confidence=vqa_result.get(
                    "confidence",
                    0.0,
                ),
                evidence=vqa_result.get(
                    "evidence",
                    [],
                ),
                outputs={
                    "image_count": len(images),
                    "query": query,
                    "metadata": metadata,
                    "image_path": image_path,
                    "device": vqa_result.get(
                        "device",
                        "unknown",
                    ),
                },
            )

            # Preserve model error information if one occurred.
            if "error" in vqa_result:
                result.outputs["error"] = vqa_result["error"]

            return asdict(result)

        except Exception as e:

            result = ToolResult(
                task=task,
                specialist=specialist,
                answer="VQA execution failed.",
                confidence=0.0,
                evidence=[],
                outputs={
                    "image_count": len(images),
                    "query": query,
                    "metadata": metadata,
                    "error": str(e),
                },
            )

            return asdict(result)

    # ---------------------------------------------------------
    # Other specialists - not connected yet
    # ---------------------------------------------------------

    result = ToolResult(
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

    return asdict(result)