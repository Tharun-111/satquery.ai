from agent.router import route_query

from models.vqa_model import answer_question
from models.grounding_model import ground_region


def execute_query(
    query: str,
    image_paths: list,
    modalities: list
) -> dict:
    """
    Main execution engine for SatQuery AI.

    The agent:
    1. Understands the user query.
    2. Routes the query to a specialist.
    3. Validates the required number of inputs.
    4. Executes the selected specialist.
    5. Returns an auditable execution result.
    """

    # =========================================================
    # 1. BASIC INPUT VALIDATION
    # =========================================================

    if not query or not query.strip():

        return {
            "query": query,
            "status": "failed",
            "task": None,
            "specialist": None,
            "reason": "Query is empty.",
            "router_confidence": 0.0,
            "inputs": image_paths,
            "modalities": modalities,
            "output": {
                "message": "Please enter a natural-language query."
            }
        }

    if not image_paths:

        return {
            "query": query,
            "status": "failed",
            "task": None,
            "specialist": None,
            "reason": "No input image supplied.",
            "router_confidence": 0.0,
            "inputs": [],
            "modalities": modalities,
            "output": {
                "message": "At least one image is required."
            }
        }

    # =========================================================
    # 2. AGENTIC ROUTING
    # =========================================================

    decision = route_query(
        query,
        len(image_paths),
        modalities
    )

    # =========================================================
    # 3. CREATE AUDITABLE EXECUTION RECORD
    # =========================================================

    result = {
        "query": query,
        "status": "started",

        "task": decision.task,

        "specialist": decision.specialist,

        "reason": decision.reason,

        "router_confidence": decision.confidence,

        "required_inputs": decision.required_inputs,

        "required_modalities": decision.required_modalities,

        "inputs": image_paths,

        "modalities": modalities,

        "output": None,
    }

    # =========================================================
    # 4. CHECK REQUIRED NUMBER OF INPUTS
    # =========================================================

    if len(image_paths) < decision.required_inputs:

        result["status"] = "failed"

        result["output"] = {
            "message": (
                f"This task requires "
                f"{decision.required_inputs} image(s), "
                f"but only {len(image_paths)} image(s) "
                f"were supplied."
            )
        }

        return result

    # =========================================================
    # 5. SINGLE IMAGE VQA
    # =========================================================

    if decision.task == "vqa":

        vqa_result = answer_question(
            image_paths[0],
            query
        )

        result["status"] = (
            "completed"
            if vqa_result.get("success", False)
            else "failed"
        )

        result["output"] = vqa_result

        return result

    # =========================================================
    # 6. TEXT-GUIDED GROUNDING
    # =========================================================

    if decision.task == "grounding":

        grounding_result = ground_region(
            image_paths[0],
            query
        )

        result["status"] = (
            "completed"
            if grounding_result.get("success", False)
            else "failed"
        )

        result["output"] = grounding_result

        return result

    # =========================================================
    # 7. BI-TEMPORAL CHANGE ANALYSIS
    # =========================================================

    if decision.task in {
        "change_detection",
        "change_vqa",
        "change_analysis"
    }:

        if len(image_paths) < 2:

            result["status"] = "failed"

            result["output"] = {
                "message": (
                    "Bi-temporal change analysis requires "
                    "two spatially corresponding images."
                )
            }

            return result

        result["status"] = "pending"

        result["output"] = {
            "message": (
                "Bi-temporal change specialist selected. "
                "Change analysis module will process the "
                "before and after images."
            ),
            "before_image": image_paths[0],
            "after_image": image_paths[1],
        }

        return result

    # =========================================================
    # 8. OPTICAL + SAR CROSS-MODAL ANALYSIS
    # =========================================================

    if decision.task == "cross_modal_analysis":

        if len(image_paths) < 2:

            result["status"] = "failed"

            result["output"] = {
                "message": (
                    "Cross-modal analysis requires an "
                    "optical image and a SAR image."
                )
            }

            return result

        result["status"] = "pending"

        result["output"] = {
            "message": (
                "Optical-SAR fusion specialist selected. "
                "The optical and SAR observations will be "
                "analysed jointly."
            ),
            "optical_image": image_paths[0],
            "sar_image": image_paths[1],
        }

        return result

    # =========================================================
    # 9. CAPTIONING / SCENE DESCRIPTION
    # =========================================================

    if decision.task in {
        "captioning",
        "caption",
        "scene_description"
    }:

        result["status"] = "pending"

        result["output"] = {
            "message": (
                "Remote-sensing captioning specialist "
                "selected."
            ),
            "image": image_paths[0],
        }

        return result

    # =========================================================
    # 10. UNKNOWN TASK
    # =========================================================

    result["status"] = "failed"

    result["output"] = {
        "message": (
            "No specialist workflow is currently "
            f"implemented for task: {decision.task}"
        )
    }

    return result