"""
SatQuery AI - Streamlit Application

Interactive remote-sensing AI assistant with:

- Image validation
- Raster preview
- Deterministic agent routing
- VQA specialist
- Grounding specialist
- Evidence generation
- Auditable execution trace
"""

import os

import streamlit as st

from agent.validator import validate_image
from agent.controller import execute_query
from processing.image_processor import analyze_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🛰️ SatQuery AI")

st.subheader(
    "Interactive Vision-Language Assistant for Remote Sensing"
)

st.markdown(
    """
SatQuery AI is an agentic remote-sensing assistant.

Upload satellite imagery, ask a natural-language question,
and the system will:

1. Validate the image
2. Analyze the imagery
3. Select the appropriate specialist
4. Run the specialist model
5. Generate auditable evidence
"""
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_path" not in st.session_state:
    st.session_state.uploaded_path = None

if "validation_result" not in st.session_state:
    st.session_state.validation_result = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "execution_result" not in st.session_state:
    st.session_state.execution_result = None


# ============================================================
# 1. IMAGE UPLOAD
# ============================================================

st.header("1. Upload Remote-Sensing Imagery")

uploaded_file = st.file_uploader(
    "Upload optical/multispectral or SAR imagery",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg",
    ],
)


if uploaded_file is not None:

    os.makedirs(
        "data/demo",
        exist_ok=True,
    )

    upload_path = os.path.join(
        "data",
        "demo",
        uploaded_file.name,
    )

    with open(upload_path, "wb") as file:
        file.write(
            uploaded_file.getbuffer()
        )

    st.session_state.uploaded_path = upload_path

    st.success(
        f"Image uploaded successfully: {uploaded_file.name}"
    )


# ============================================================
# STOP UNTIL IMAGE IS PROVIDED
# ============================================================

if st.session_state.uploaded_path is None:

    st.info(
        "Upload a satellite image to begin."
    )

    st.divider()

    st.caption(
        "SatQuery AI • Agentic Remote-Sensing Vision-Language Assistant"
    )

    st.stop()


image_path = st.session_state.uploaded_path


# ============================================================
# 2. IMAGE MODALITY
# ============================================================

st.header("2. Image Configuration")

modality = st.selectbox(
    "Image modality",
    [
        "optical",
        "sar",
    ],
)


# ============================================================
# 3. IMAGE VALIDATION
# ============================================================

st.subheader("Input Validation")

try:

    validation_result = validate_image(
        image_path
    )

    st.session_state.validation_result = (
        validation_result
    )

except Exception as error:

    st.error(
        f"Image validation failed: {error}"
    )

    st.stop()


if validation_result.get("valid"):

    st.success(
        "✅ Input image is valid"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Width",
            validation_result.get(
                "width",
                "N/A",
            ),
        )

    with col2:

        st.metric(
            "Height",
            validation_result.get(
                "height",
                "N/A",
            ),
        )

    with col3:

        st.metric(
            "Bands",
            validation_result.get(
                "bands",
                "N/A",
            ),
        )

    with col4:

        st.metric(
            "Georeferenced",
            (
                "Yes"
                if validation_result.get(
                    "georeferenced",
                    False,
                )
                else "No"
            ),
        )

    st.caption(
        validation_result.get(
            "message",
            "",
        )
    )

else:

    st.error(
        "❌ Input image is invalid."
    )

    st.write(
        validation_result
    )

    st.stop()


# ============================================================
# 4. IMAGE ANALYSIS
# ============================================================

st.header("3. Image Analysis")

try:

    with st.spinner(
        "🔬 Analyzing remote-sensing image..."
    ):

        analysis_result = analyze_image(
            image_path
        )

        st.session_state.analysis_result = (
            analysis_result
        )

except Exception as error:

    st.error(
        f"Image processing failed: {error}"
    )

    st.stop()


# ============================================================
# VISUAL PREVIEW
# ============================================================

if analysis_result.get("display_image") is not None:

    st.image(
        analysis_result["display_image"],
        caption="Remote-Sensing Image",
        width="stretch",
    )


# ============================================================
# IMAGE METADATA
# ============================================================

with st.expander(
    "📊 Image Metadata and Statistics"
):

    metadata = analysis_result.get(
        "metadata",
        {},
    )

    statistics = analysis_result.get(
        "statistics",
        {},
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Image Metadata"
        )

        st.json(
            metadata
        )

    with col2:

        st.subheader(
            "Band Statistics"
        )

        st.json(
            statistics
        )


# ============================================================
# 5. NATURAL LANGUAGE QUERY
# ============================================================

st.header("4. Ask SatQuery AI")

query = st.text_input(
    "Ask a question about the image",
    placeholder=(
        "Examples: Is there water? "
        "Find the water body. "
        "Locate buildings."
    ),
)


# ============================================================
# EXECUTE AGENT
# ============================================================

if st.button(
    "🤖 Analyze with SatQuery AI",
    type="primary",
    use_container_width=True,
):

    if not query.strip():

        st.warning(
            "Please enter a question first."
        )

        st.stop()

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 SatQuery AI is selecting and running the specialist..."
    ):

        try:

            execution_result = execute_query(
                query=query,
                images=[
                    image_path
                ],
                modalities=[
                    modality
                ],
                metadata=(
                    analysis_result.get(
                        "metadata",
                        {},
                    )
                ),
            )

            st.session_state.execution_result = (
                execution_result
            )

        except Exception as error:

            st.error(
                f"Agent execution failed: {error}"
            )

            st.stop()


# ============================================================
# DISPLAY EXECUTION RESULT
# ============================================================

execution_result = (
    st.session_state.execution_result
)


if execution_result is not None:

    # ========================================================
    # 6. AGENT DECISION
    # ========================================================

    st.header("5. Agent Decision")

    plan = execution_result.get(
        "plan",
        {},
    )

    result = execution_result.get(
        "result",
        {},
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Selected Task",
            str(
                plan.get(
                    "task",
                    "unknown",
                )
            ).upper(),
        )

    with col2:

        st.metric(
            "Routing Confidence",
            f"{float(plan.get('confidence', 0.0)):.0%}",
        )

    with col3:

        st.metric(
            "Input Modality",
            modality.upper(),
        )

    st.info(
        f"""
**Selected Specialist**

{plan.get("specialist", "Unknown")}

**Routing Reason**

{plan.get("reason", "No reason provided.")}
"""
    )


    # ========================================================
    # 7. AI ANSWER
    # ========================================================

    st.header("6. AI Answer")

    answer = result.get(
        "answer",
        "No answer returned.",
    )

    confidence = float(
        result.get(
            "confidence",
            0.0,
        )
    )

    st.success(
        answer
    )

    st.metric(
        "Model Confidence",
        f"{confidence:.0%}",
    )


    # ========================================================
    # 8. EVIDENCE
    # ========================================================

    st.header("7. Evidence")

    evidence = result.get(
        "evidence",
        [],
    )

    if evidence:

        for item in evidence:

            st.write(
                f"• {item}"
            )

    else:

        st.info(
            "No evidence statements were returned."
        )


    # ========================================================
    # 9. GROUNDING REGIONS
    # ========================================================

    outputs = result.get(
        "outputs",
        {},
    )

    regions = outputs.get(
        "regions",
        [],
    )

    if plan.get("task") == "grounding":

        st.header("8. Grounding Results")

        if regions:

            st.success(
                f"{len(regions)} region(s) detected."
            )

            for index, region in enumerate(
                regions,
                start=1,
            ):

                st.write(
                    f"**Region {index}**"
                )

                st.json(
                    region
                )

        else:

            st.info(
                "No matching region was detected "
                "in this image."
            )


    # ========================================================
    # 10. EXECUTION TRACE
    # ========================================================

    st.header("9. Agent Execution Trace")

    execution_trace = {

        "success": execution_result.get(
            "success",
            False,
        ),

        "query": query,

        "input_count": len(
            [image_path]
        ),

        "input_modality": modality,

        "selected_task": plan.get(
            "task"
        ),

        "selected_specialist": plan.get(
            "specialist"
        ),

        "routing_confidence": plan.get(
            "confidence"
        ),

        "input_validation": (
            "PASSED"
            if plan.get(
                "valid",
                False,
            )
            else "FAILED"
        ),

        "validation_message": plan.get(
            "validation_message"
        ),

        "model_device": outputs.get(
            "device",
            "unknown",
        ),

        "visual_evidence": (
            "GENERATED"
            if evidence
            else "NONE"
        ),
    }

    st.json(
        execution_trace
    )


    # ========================================================
    # 11. RAW RESULT
    # ========================================================

    with st.expander(
        "🔧 Raw Agent Result"
    ):

        st.json(
            execution_result
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SatQuery AI • Agentic Remote-Sensing Vision-Language Assistant"
)