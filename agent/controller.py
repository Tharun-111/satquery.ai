"""
SatQuery AI - Agent Controller

Connects the deterministic router with the specialist
model registry and produces an auditable execution plan.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from agent.router import route_query
from agent.model_registry import get_tool


@dataclass
class ExecutionPlan:
    task: str
    specialist: str
    reason: str
    confidence: float
    image_count: int
    modalities: List[str]
    valid: bool
    validation_message: str


def _validate_tool_inputs(tool, image_count: int, modalities: List[str]):
    """
    Check whether the selected specialist can accept the
    supplied number and type of images.
    """

    if image_count < tool.input_count:
        return False, (
            f"{tool.name} requires {tool.input_count} image(s), "
            f"but only {image_count} were supplied."
        )

    if tool.task == "cross_modal_analysis":

        normalized = [m.lower() for m in modalities]

        if "optical" not in normalized and "multispectral" not in normalized:
            return False, "Optical/multispectral imagery is required."

        if "sar" not in normalized:
            return False, "SAR imagery is required."

    return True, "Input configuration is compatible with the selected specialist."


def create_execution_plan(
    query: str,
    image_count: int = 1,
    modalities=None,
) -> Dict[str, Any]:

    if modalities is None:
        modalities = []

    # ---------------------------------------------
    # Step 1: Route the query
    # ---------------------------------------------

    decision = route_query(
        query=query,
        image_count=image_count,
        modalities=modalities,
    )

    # ---------------------------------------------
    # Step 2: Find specialist tool
    # ---------------------------------------------

    tool = get_tool(decision.task)

    # ---------------------------------------------
    # Step 3: Validate inputs
    # ---------------------------------------------

    valid, message = _validate_tool_inputs(
        tool=tool,
        image_count=image_count,
        modalities=modalities,
    )

    # ---------------------------------------------
    # Step 4: Build auditable execution plan
    # ---------------------------------------------

    plan = ExecutionPlan(
        task=decision.task,
        specialist=tool.name,
        reason=decision.reason,
        confidence=decision.confidence,
        image_count=image_count,
        modalities=modalities,
        valid=valid,
        validation_message=message,
    )

    return asdict(plan)