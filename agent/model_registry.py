"""
SatQuery AI - Specialist Model Registry

Central registry of all specialist models/tools used by
the agentic controller.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SpecialistTool:
    name: str
    task: str
    description: str
    input_count: int
    modalities: List[str]
    enabled: bool = True


MODEL_REGISTRY: Dict[str, SpecialistTool] = {

    "vqa": SpecialistTool(
        name="Remote-Sensing VQA Tool",
        task="vqa",
        description="Answers natural-language questions about a remote-sensing image.",
        input_count=1,
        modalities=["optical", "multispectral", "sar"],
    ),

    "captioning": SpecialistTool(
        name="Remote-Sensing Captioning Tool",
        task="captioning",
        description="Generates a scene description or caption for a remote-sensing image.",
        input_count=1,
        modalities=["optical", "multispectral", "sar"],
    ),

    "grounding": SpecialistTool(
        name="Remote-Sensing Grounding Tool",
        task="grounding",
        description="Locates and highlights a region referred to by the user's text query.",
        input_count=1,
        modalities=["optical", "multispectral", "sar"],
    ),

    "change_analysis": SpecialistTool(
        name="Temporal Change Understanding Tool",
        task="change_analysis",
        description="Analyzes two spatially corresponding images acquired at different times.",
        input_count=2,
        modalities=["optical", "multispectral", "sar"],
    ),

    "cross_modal_analysis": SpecialistTool(
        name="Optical-SAR Fusion Analysis Tool",
        task="cross_modal_analysis",
        description="Combines complementary information from co-registered optical and SAR imagery.",
        input_count=2,
        modalities=["optical", "sar"],
    ),
}


def get_tool(task: str) -> SpecialistTool:
    """Return the specialist registered for a task."""

    if task not in MODEL_REGISTRY:
        raise ValueError(f"No specialist registered for task: {task}")

    return MODEL_REGISTRY[task]


def list_tools() -> List[SpecialistTool]:
    """Return all registered specialist tools."""

    return list(MODEL_REGISTRY.values())