import os
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from agent.validator import validate_image
from agent.router import route_query
from models.vqa_model import answer_question
from models.caption_model import create_caption_model
from models.disaster_model import create_disaster_model
from models.grounding.grounding_model import create_grounding_model
from processing.change_detectoe import analyze_change


st.set_page_config(
    page_title="SatQuery AI",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ SatQuery AI")
st.subheader(
    "Interactive Vision-Language Assistant for Remote Sensing"
)

st.markdown(
    """
    Upload remote-sensing imagery and ask questions in natural language.
    SatQuery AI validates the inputs, identifies the task, selects a
    specialist workflow, and returns evidence-grounded results.
    """
)

st.divider()


# ============================================================
# 1. INPUT MODE
# ============================================================

st.header("1. Select Analysis Type")

analysis_mode = st.radio(
    "Choose the type of analysis",
    [
        "Single Image",
        "Object Detection",
        "Disaster Analysis",
        "Bi-Temporal Change Analysis"
    ],
    horizontal=True
)


# ============================================================
# 2. SINGLE IMAGE
# ============================================================

if analysis_mode in [
    "Single Image",
    "Object Detection",
    "Disaster Analysis"
]:

    st.header(
        "2. Upload Remote-Sensing Image"
        if analysis_mode == "Single Image"
        else f"2. Upload Image for {analysis_mode}"
    )

    if analysis_mode == "Object Detection":
        st.info(
            "Ask where objects such as buildings, roads, or water are "
            "located in the image."
        )
    elif analysis_mode == "Disaster Analysis":
        st.info(
            "Ask about visible damage, affected areas, flooding, fires, "
            "or other disaster-related evidence."
        )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        key="single_image"
    )

    image_path = None
    validation_result = None

    if uploaded_file is not None:

        image_path = "data/demo/" + uploaded_file.name

        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        validation_result = validate_image(
            image_path
        )

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

        else:

            st.error(
                "❌ Input image is invalid"
            )

            st.write(
                validation_result["message"]
            )

        if uploaded_file.name.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            st.image(
                uploaded_file,
                caption="Uploaded Remote-Sensing Image",
                width="stretch"
            )

    st.divider()

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    st.header("3. Ask SatQuery AI")

    query = st.text_area(
        "Enter your natural-language query",
        placeholder=(
            "Example: Describe the land-cover and major "
            "objects visible in this image."
            if analysis_mode == "Single Image"
            else (
                "Example: Find buildings, roads, or water in this image."
                if analysis_mode == "Object Detection"
                else (
                    "Example: Identify visible flood damage or affected "
                    "areas in this image."
                )
            )
        ),
        height=100,
        key="single_query"
    )

    if st.button(
        "🚀 Analyze Image",
        width="stretch",
        key="single_analyze"
    ):

        if uploaded_file is None:

            st.warning(
                "⚠️ Please upload an image first."
            )

        elif (
            validation_result is None
            or not validation_result["valid"]
        ):

            st.error(
                "❌ The uploaded image failed validation."
            )

        elif not query.strip():

            st.warning(
                "⚠️ Please enter a query."
            )

        else:

            # Determine basic modality
            if validation_result["bands"] >= 3:
                modality = "optical"
            else:
                modality = "sar"

            # Agent routing
            decision = route_query(
                query,
                1,
                [modality]
            )

            st.divider()

            st.header("4. Agentic Execution")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Selected Task",
                    decision.task.upper()
                )

            with col2:
                st.metric(
                    "Specialist",
                    decision.specialist
                )

            with col3:
                st.metric(
                    "Confidence",
                    f"{decision.confidence:.0%}"
                )

            st.info(
                f"**Routing Reason:** {decision.reason}"
            )

            # ------------------------------------------------
            # DISASTER ANALYSIS
            # ------------------------------------------------

            if analysis_mode == "Disaster Analysis":
                st.subheader("🚨 Disaster Analysis")

                with st.spinner("Running CLIP disaster classification..."):
                    disaster_model = create_disaster_model()
                    result = disaster_model.analyze(
                        image_path,
                        query
                    )

                st.success("Disaster analysis completed.")
                st.write(result["answer"])
                st.progress(int(result["confidence"] * 100))
                st.subheader("Classification Evidence")
                st.json(result["classifications"])

            # ------------------------------------------------
            # IMAGE CAPTIONING
            # ------------------------------------------------

            elif decision.task == "captioning":
                st.subheader("📝 Remote-Sensing Image Caption")

                with st.spinner("Running BLIP image captioning..."):
                    caption_model = create_caption_model()
                    result = caption_model.describe(
                        image_path,
                        query
                    )

                st.success("Image description completed.")
                st.write(result["answer"])
                st.progress(int(result["confidence"] * 100))

                st.subheader("Caption Evidence")
                for item in result["evidence"]:
                    st.write(f"• {item}")

            # ------------------------------------------------
            # VQA
            # ------------------------------------------------

            elif decision.task == "vqa":

                st.subheader(
                    "🧠 Remote-Sensing VQA"
                )

                with st.spinner(
                    "Analysing image..."
                ):

                    result = answer_question(
                        image_path,
                        query
                    )

                st.success(
                    "Analysis completed."
                )

                st.markdown("### Answer")

                st.write(
                    result["answer"]
                )

                st.markdown(
                    "### Confidence"
                )

                st.progress(
                    int(
                        result["confidence"] * 100
                    )
                )

                st.write(
                    f"{result['confidence']:.0%}"
                )

                st.markdown(
                    "### Evidence"
                )

                for item in result["evidence"]:

                    st.write(
                        f"• {item}"
                    )

            # ------------------------------------------------
            # GROUNDING
            # ------------------------------------------------

            elif (
                analysis_mode == "Object Detection"
                or decision.task == "grounding"
            ):

                st.subheader(
                    "📍 Remote-Sensing Grounding"
                )

                st.info(
                    "The agent selected the grounding workflow "
                    "because the query requests a spatial region."
                )

                with st.spinner("Running Grounding DINO..."):
                    grounding_model = create_grounding_model()
                    result = grounding_model.ground(
                        image_path,
                        query
                    )

                st.success("Object detection completed.")
                st.write(result["answer"])
                st.progress(int(result["confidence"] * 100))

                count_label = "Detected Objects"
                if "building" in query.lower():
                    count_label = "Buildings Detected"

                st.metric(
                    count_label,
                    result.get(
                        "requested_count",
                        result.get("detected_count", 0)
                    )
                )

                if result.get("regions"):
                    st.subheader("Detected Objects")
                    st.json(result["regions"])
                else:
                    st.info("No matching objects were detected.")

            else:

                st.warning(
                    f"The selected task is `{decision.task}` "
                    "but its specialist is not connected yet."
                )

            # ------------------------------------------------
            # EXECUTION TRACE
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📋 Execution Trace"
            )

            st.json(
                {
                    "selected_task": decision.task,
                    "specialist": decision.specialist,
                    "routing_confidence": decision.confidence,
                    "input_file": uploaded_file.name,
                    "input_modality": modality,
                    "required_inputs": decision.required_inputs,
                    "required_modalities":
                        decision.required_modalities
                }
            )


# ============================================================
# 3. BI-TEMPORAL CHANGE ANALYSIS
# ============================================================

else:

    st.header(
        "2. Upload Bi-Temporal Images"
    )

    st.info(
        "Upload two spatially corresponding images: "
        "a BEFORE image and an AFTER image."
    )

    col1, col2 = st.columns(2)

    before_file = None
    after_file = None

    with col1:

        st.subheader(
            "🕐 Image 1 — BEFORE"
        )

        before_file = st.file_uploader(
            "Upload BEFORE image",
            type=[
                "tif",
                "tiff",
                "png",
                "jpg",
                "jpeg"
            ],
            key="before_image"
        )

    with col2:

        st.subheader(
            "🕐 Image 2 — AFTER"
        )

        after_file = st.file_uploader(
            "Upload AFTER image",
            type=[
                "tif",
                "tiff",
                "png",
                "jpg",
                "jpeg"
            ],
            key="after_image"
        )

    before_path = None
    after_path = None

    if before_file is not None:

        before_path = (
            "data/demo/before_"
            + before_file.name
        )

        with open(before_path, "wb") as f:
            f.write(
                before_file.getbuffer()
            )

        st.success(
            f"Before image loaded: {before_file.name}"
        )

    if after_file is not None:

        after_path = (
            "data/demo/after_"
            + after_file.name
        )

        with open(after_path, "wb") as f:
            f.write(
                after_file.getbuffer()
            )

        st.success(
            f"After image loaded: {after_file.name}"
        )

    # --------------------------------------------------------
    # SHOW IMAGES
    # --------------------------------------------------------

    if before_file is not None:

        with col1:

            if before_file.name.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                st.image(
                    before_file,
                    caption="BEFORE",
                    width="stretch"
                )

    if after_file is not None:

        with col2:

            if after_file.name.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                st.image(
                    after_file,
                    caption="AFTER",
                    width="stretch"
                )

    st.divider()

    # --------------------------------------------------------
    # CHANGE QUERY
    # --------------------------------------------------------

    st.header(
        "3. Ask About the Change"
    )

    change_query = st.text_area(
        "Enter your change-analysis question",
        placeholder=(
            "Example: What changed between these two dates, "
            "and where did the change occur?"
        ),
        height=100,
        key="change_query"
    )

    if st.button(
        "🔄 Analyze Change",
        width="stretch",
        key="change_analyze"
    ):

        if before_file is None:

            st.warning(
                "⚠️ Please upload the BEFORE image."
            )

        elif after_file is None:

            st.warning(
                "⚠️ Please upload the AFTER image."
            )

        elif not change_query.strip():

            st.warning(
                "⚠️ Please enter a change-analysis query."
            )

        else:

            # ------------------------------------------------
            # AGENT ROUTING
            # ------------------------------------------------

            decision = route_query(
                change_query,
                2,
                ["optical", "optical"]
            )

            st.divider()

            st.header(
                "4. Agentic Change Workflow"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Selected Task",
                    decision.task.upper()
                )

            with col2:

                st.metric(
                    "Specialist",
                    decision.specialist
                )

            with col3:

                st.metric(
                    "Confidence",
                    f"{decision.confidence:.0%}"
                )

            st.info(
                f"**Routing Reason:** {decision.reason}"
            )

            # ------------------------------------------------
            # RUN CHANGE DETECTOR
            # ------------------------------------------------

            with st.spinner(
                "Comparing the two observations..."
            ):

                try:

                    result = analyze_change(
                        before_path,
                        after_path
                    )

                    st.success(
                        "✅ Change analysis completed."
                    )

                    # ------------------------------------------------
                    # METRICS
                    # ------------------------------------------------

                    st.subheader(
                        "📊 Change Summary"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Changed Area",
                            f"{result['change_percentage']:.2f}%"
                        )

                    with col2:

                        st.metric(
                            "Changed Pixels",
                            f"{result['changed_pixels']:,}"
                        )

                    with col3:

                        st.metric(
                            "Total Pixels",
                            f"{result['total_pixels']:,}"
                        )

                    # ------------------------------------------------
                    # TEXT RESULT
                    # ------------------------------------------------

                    st.subheader(
                        "🧠 Interpretation"
                    )

                    st.write(
                        result["summary"]
                    )

                    # ------------------------------------------------
                    # VISUAL EVIDENCE
                    # ------------------------------------------------

                    st.subheader(
                        "🗺️ Visual Change Evidence"
                    )

                    overlay = result["overlay"]

                    st.image(
                        overlay,
                        caption=(
                            "Detected changes highlighted "
                            "on the AFTER image"
                        ),
                        width="stretch"
                    )

                    # ------------------------------------------------
                    # DOWNLOAD EVIDENCE
                    # ------------------------------------------------

                    output_file = (
                        "outputs/change_evidence.png"
                    )

                    os.makedirs(
                        "outputs",
                        exist_ok=True
                    )

                    from PIL import Image

                    Image.fromarray(overlay).save(
                        output_file,
                        format="PNG"
                    )

                    with open(
                        output_file,
                        "rb"
                    ) as f:

                        st.download_button(
                            "⬇️ Download Change Evidence",
                            f,
                            file_name="change_evidence.png",
                            mime="image/png"
                        )

                except Exception as exc:

                    st.error(
                        f"Change analysis failed: {exc}"
                    )

            # ------------------------------------------------
            # EXECUTION TRACE
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "📋 Execution Trace"
            )

            st.json(
                {
                    "selected_task": decision.task,
                    "specialist": decision.specialist,
                    "routing_confidence":
                        decision.confidence,
                    "input_count": 2,
                    "before_image":
                        before_file.name,
                    "after_image":
                        after_file.name,
                    "workflow":
                        "bi-temporal-change-analysis"
                }
            )