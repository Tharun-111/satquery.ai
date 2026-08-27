"""
SatQuery AI - Agent Controller

Connects the deterministic router with the specialist
model registry and produces an auditable execution plan.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from agent.router import route_query
from agent.model_registry import get_tool
from agent.executor import execute_specialist


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
def execute_query(
    query: str,
    images=None,
    modalities=None,
    metadata=None,
) -> Dict[str, Any]:
    """
    Complete SatQuery AI execution pipeline.

    Steps:
        1. Count uploaded images
        2. Create execution plan
        3. Validate inputs
        4. Execute selected specialist
        5. Return auditable result
    """

    if images is None:
        images = []

    if modalities is None:
        modalities = []

    if metadata is None:
        metadata = {}

    # ---------------------------------------------------------
    # Step 1: Create execution plan
    # ---------------------------------------------------------

    plan = create_execution_plan(
        query=query,
        image_count=len(images),
        modalities=modalities,
    )

    # ---------------------------------------------------------
    # Step 2: Stop if inputs are invalid
    # ---------------------------------------------------------

    if not plan["valid"]:
        return {
            "success": False,
            "plan": plan,
            "result": {
                "task": plan["task"],
                "specialist": plan["specialist"],
                "answer": "Input validation failed.",
                "confidence": 0.0,
                "evidence": [],
                "outputs": {
                    "image_count": len(images),
                    "query": query,
                    "metadata": metadata,
                },
            },
        }

    # ---------------------------------------------------------
    # Step 3: Execute specialist
    # ---------------------------------------------------------

    result = execute_specialist(
        task=plan["task"],
        specialist=plan["specialist"],
        images=images,
        query=query,
        metadata=metadata,
    )

    # ---------------------------------------------------------
    # Step 4: Return complete auditable response
    # ---------------------------------------------------------

    return {
        "success": True,
        "plan": plan,
        "result": result,
    }