"""
SatQuery AI - Agent Router

Routes a natural-language remote-sensing query to the
appropriate specialist workflow.

The router is intentionally deterministic so that the
execution trace is auditable during evaluation.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RouteDecision:
    task: str
    specialist: str
    reason: str
    required_inputs: int
    required_modalities: List[str]
    confidence: float


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def route_query(query: str, image_count: int = 1, modalities=None) -> RouteDecision:
    """
    Select the appropriate SatQuery AI workflow.

    Parameters
    ----------
    query:
        Natural-language user query.

    image_count:
        Number of uploaded images.

    modalities:
        Optional list such as ["optical"], ["sar"],
        ["optical", "sar"] or ["optical", "optical"].
    """

    query_lower = query.lower().strip()

    if modalities is None:
        modalities = []

    # ---------------------------------------------------------
    # 1. Bi-temporal / CHANGE analysis
    # ---------------------------------------------------------

    change_keywords = [
        "change",
        "changed",
        "difference",
        "before and after",
        "between these dates",
        "between the two",
        "increased",
        "decreased",
        "new building",
        "removed",
        "demolished",
        "growth",
        "loss",
        "temporal",
    ]

    if image_count >= 2 and _contains_any(query_lower, change_keywords):

        return RouteDecision(
            task="change_analysis",
            specialist="Temporal Change Understanding Tool",
            reason="The query requests comparison/change understanding and multiple images were supplied.",
            required_inputs=2,
            required_modalities=["compatible_temporal_pair"],
            confidence=0.96,
        )

    # ---------------------------------------------------------
    # 2. Optical + SAR cross-modal analysis
    # ---------------------------------------------------------

    cross_modal_keywords = [
        "sar",
        "radar",
        "optical and sar",
        "optical + sar",
        "optical with sar",
        "both images",
        "cross-modal",
        "cross modal",
        "multimodal",
        "built-up and water",
    ]

    has_optical = any(
        modality.lower() in ["optical", "multispectral"]
        for modality in modalities
    )

    has_sar = any(
        modality.lower() == "sar"
        for modality in modalities
    )

    if image_count >= 2 and (
        _contains_any(query_lower, cross_modal_keywords)
        or (has_optical and has_sar)
    ):

        return RouteDecision(
            task="cross_modal_analysis",
            specialist="Optical-SAR Fusion Analysis Tool",
            reason="Multiple images with optical/SAR or cross-modal analysis intent were supplied.",
            required_inputs=2,
            required_modalities=["optical", "sar"],
            confidence=0.95,
        )

    # ---------------------------------------------------------
    # 3. Text-guided region grounding
    # ---------------------------------------------------------

    grounding_keywords = [
        "highlight",
        "locate",
        "where is",
        "where are",
        "find",
        "identify the location",
        "mark",
        "region",
        "bounding box",
        "ground",
    ]

    if _contains_any(query_lower, grounding_keywords):

        return RouteDecision(
            task="grounding",
            specialist="Remote-Sensing Grounding Tool",
            reason="The query asks the system to locate, highlight, or identify a spatial region.",
            required_inputs=1,
            required_modalities=["optical_or_sar"],
            confidence=0.91,
        )

    # ---------------------------------------------------------
    # 4. Captioning / scene description
    # ---------------------------------------------------------

    caption_keywords = [
        "describe",
        "description",
        "caption",
        "scene",
        "land-cover",
        "land cover",
        "major objects",
        "what is visible",
        "what can you see",
    ]

    if _contains_any(query_lower, caption_keywords):

        return RouteDecision(
            task="captioning",
            specialist="Remote-Sensing Captioning Tool",
            reason="The query requests a scene description or image caption.",
            required_inputs=1,
            required_modalities=["optical_or_sar"],
            confidence=0.92,
        )

    # ---------------------------------------------------------
    # 5. Default = Visual Question Answering
    # ---------------------------------------------------------

    return RouteDecision(
        task="vqa",
        specialist="Remote-Sensing VQA Tool",
        reason="The query is treated as a visual question requiring an answer from the image.",
        required_inputs=1,
        required_modalities=["optical_or_sar"],
        confidence=0.85,
    )