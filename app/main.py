import os
import streamlit as st

from agent.validator import validate_image
from agent.router import route_query
from processing.image_processor import analyze_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
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
    and the system validates the image, analyzes the imagery,
    selects the appropriate specialist workflow, and produces
    evidence for the final answer.
    """
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "validation_result" not in st.session_state:
    st.session_state.validation_result = None

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "route_result" not in st.session_state:
    st.session_state.route_result = None


# ============================================================
# 1. IMAGE UPLOAD
# ============================================================

st.header("1. Upload Remote-Sensing Imagery")

uploaded_file = st.file_uploader(
    "Upload optical/multispectral or SAR image",
    type=[
        "tif",
        "tiff",
        "png",
        "jpg",
        "jpeg"
    ]
)


if uploaded_file is not None:

    # Create demo directory
    os.makedirs(
        "data/demo",
        exist_ok=True
    )

    # Save uploaded image
    upload_path = os.path.join(
        "data",
        "demo",
        uploaded_file.name
    )

    with open(upload_path, "wb") as file:
        file.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"Image uploaded successfully: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # IMAGE MODALITY
    # --------------------------------------------------------

    modality = st.selectbox(
        "Image modality",
        [
            "optical",
            "sar"
        ]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_result = validate_image(
        upload_path
    )

    st.session_state.validation_result = (
        validation_result
    )

    st.subheader("Input Validation")

    if validation_result["valid"]:

        st.success(
            "✅ Input image is valid"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Width",
                validation_result["width"]
            )

        with col2:
            st.metric(
                "Height",
                validation_result["height"]
            )

        with col3:
            st.metric(
                "Bands",
                validation_result["bands"]
            )

        with col4:
            st.metric(
                "Georeferenced",
                "Yes"
                if validation_result["georeferenced"]
                else "No"
            )

        st.caption(
            validation_result["message"]
        )

    else:

        st.error(
            "❌ Input image is invalid"
        )

        st.write(
            validation_result["message"]
        )

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    if uploaded_file.name.lower().endswith(
        (
            ".png",
            ".jpg",
            ".jpeg"
        )
    ):

        st.image(
            uploaded_file,
            caption="Uploaded Remote-Sensing Image",
            width="stretch"
        )


st.divider()


# ============================================================
# 2. NATURAL LANGUAGE QUERY
# ============================================================

st.header("2. Ask SatQuery AI")

query = st.text_area(
    "Enter your natural-language query",

    placeholder=(
        "Examples:\n"
        "Describe the land-cover and major objects visible in this image.\n"
        "How many buildings are visible?\n"
        "Highlight the water body in this image."
    ),

    height=130
)


# ============================================================
# 3. ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🚀 Analyze Image",
    type="primary",
    width="stretch"
)


if analyze_button:

    # --------------------------------------------------------
    # CHECK IMAGE
    # --------------------------------------------------------

    if uploaded_file is None:

        st.warning(
            "Please upload an image first."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK QUERY
    # --------------------------------------------------------

    if not query.strip():

        st.warning(
            "Please enter a natural-language query."
        )

        st.stop()


    # --------------------------------------------------------
    # CHECK VALIDATION
    # --------------------------------------------------------

    validation_result = (
        st.session_state.validation_result
    )

    if (
        validation_result is None
        or not validation_result["valid"]
    ):

        st.error(
            "The uploaded image did not pass validation."
        )

        st.stop()


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    with st.spinner(
        "🔍 Processing remote-sensing image..."
    ):

        try:

            analysis_result = analyze_image(
                upload_path
            )

            st.session_state.analysis_result = (
                analysis_result
            )

        except Exception as error:

            st.error(
                f"Image processing failed: {error}"
            )

            st.stop()


    # ========================================================
    # AGENT ROUTING
    # ========================================================

    with st.spinner(
        "🤖 SatQuery AI is selecting the specialist..."
    ):

        try:

            # IMPORTANT:
            # route_query() expects positional arguments:
            #
            # route_query(query, number_of_inputs, modalities)
            #
            # Do NOT use num_inputs= here.

            route_result = route_query(
                query,
                1,
                [modality]
            )

            st.session_state.route_result = (
                route_result
            )

        except Exception as error:

            st.error(
                f"Agent routing failed: {error}"
            )

            st.stop()


    # ========================================================
    # SUCCESS
    # ========================================================

    st.success(
        "✅ SatQuery AI completed the initial analysis pipeline."
    )


    # ========================================================
    # 4. AGENT DECISION
    # ========================================================

    st.header("3. Agent Decision")

    route_result = (
        st.session_state.route_result
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Selected Task",
            route_result.task.upper()
        )

    with col2:

        st.metric(
            "Confidence",
            f"{route_result.confidence:.0%}"
        )

    with col3:

        st.metric(
            "Input Modality",
            modality.upper()
        )


    st.info(
        f"""
        **Selected Specialist:**  
        {route_result.specialist}

        **Routing Reason:**  
        {route_result.reason}
        """
    )


    # ========================================================
    # 5. VISUAL EVIDENCE
    # ========================================================

    st.header("4. Visual Evidence")

    analysis_result = (
        st.session_state.analysis_result
    )

    if analysis_result.get(
        "display_image"
    ) is not None:

        st.image(
            analysis_result["display_image"],
            caption="Processed Remote-Sensing Image",
            width="stretch"
        )


    # ========================================================
    # 6. IMAGE METADATA
    # ========================================================

    st.header("5. Image Evidence")

    metadata = (
        analysis_result["metadata"]
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
            analysis_result["statistics"]
        )


    # ========================================================
    # 7. EXECUTION TRACE
    # ========================================================

    st.header("6. Agent Execution Trace")

    execution_trace = {

        "query": query,

        "input_count": 1,

        "input_modality": modality,

        "selected_task": route_result.task,

        "selected_specialist": (
            route_result.specialist
        ),

        "routing_confidence": (
            route_result.confidence
        ),

        "image_processor": (
            "SatQuery Image Processing Engine"
        ),

        "input_validation": "PASSED",

        "visual_evidence": "GENERATED"

    }

    st.json(
        execution_trace
    )


    # ========================================================
    # 8. SPECIALIST STATUS
    # ========================================================

    st.header("7. Specialist Workflow")

    if route_result.task == "vqa":

        st.info(
            """
            🧠 **Remote-Sensing VQA Specialist Selected**

            The agent identified this request as a
            visual question-answering task.

            Image evidence has been prepared for the
            remote-sensing VQA model.
            """
        )

    elif route_result.task == "grounding":

        st.info(
            """
            🎯 **Remote-Sensing Grounding Specialist Selected**

            The agent identified this request as a
            text-guided spatial grounding task.

            Image evidence has been prepared for the
            grounding model.
            """
        )

    elif route_result.task == "captioning":

        st.info(
            """
            📝 **Remote-Sensing Captioning Specialist Selected**

            The agent identified this request as an
            image captioning / scene description task.
            """
        )

    elif route_result.task == "change_analysis":

        st.warning(
            """
            🔄 **Change Analysis Selected**

            Change analysis requires two spatially
            corresponding images acquired at different
            times.

            The current single-image input cannot execute
            the change workflow yet.
            """
        )

    elif route_result.task == "cross_modal_analysis":

        st.warning(
            """
            🛰️ **Optical-SAR Fusion Selected**

            Cross-modal analysis requires two images:

            1. Optical / multispectral
            2. SAR

            The current interface contains only one image.
            """
        )

    else:

        st.info(
            "Specialist workflow selected successfully."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SatQuery AI • Agentic Remote-Sensing Vision-Language Assistant"
)