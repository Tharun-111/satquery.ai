import os
import streamlit as st

from agent.validator import validate_image
from processing.image_processor import analyze_image
from agent.controller import execute_query


# ============================================================
# PAGE CONFIG
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
    Upload satellite imagery, ask a natural-language question,
    and SatQuery AI will validate the image, process the imagery,
    select the appropriate specialist, run the model, and produce
    auditable evidence.
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

if "query_result" not in st.session_state:
    st.session_state.query_result = None


# ============================================================
# 1. IMAGE UPLOAD
# ============================================================

st.header("1. Upload Remote-Sensing Imagery")

uploaded_file = st.file_uploader(
    "Upload optical/multispectral or SAR image",
    type=["tif", "tiff", "png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    os.makedirs("data/demo", exist_ok=True)

    upload_path = os.path.join(
        "data",
        "demo",
        uploaded_file.name
    )

    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(
        f"Image uploaded successfully: {uploaded_file.name}"
    )


    # ========================================================
    # MODALITY
    # ========================================================

    modality = st.selectbox(
        "Image modality",
        ["optical", "sar"]
    )


    # ========================================================
    # 2. VALIDATION
    # ========================================================

    st.header("2. Input Validation")

    try:

        validation_result = validate_image(upload_path)

        st.session_state.validation_result = validation_result

    except Exception as error:

        st.error(
            f"Image validation failed: {error}"
        )

        st.stop()


    if validation_result["valid"]:

        st.success("✅ Input image is valid")

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
            validation_result.get(
                "message",
                "Invalid image."
            )
        )

        st.stop()


    # ========================================================
    # 3. IMAGE PROCESSING
    # ========================================================

    st.header("3. Remote-Sensing Image")

    try:

        with st.spinner("🛰️ Processing remote-sensing image..."):

            analysis_result = analyze_image(
                upload_path
            )

        st.session_state.analysis_result = analysis_result

    except Exception as error:

        st.error(
            f"Image processing failed: {error}"
        )

        st.stop()


    # Display image
    display_image = analysis_result.get("display_image")

    if display_image is not None:

        st.image(
            display_image,
            caption="Processed Remote-Sensing Image",
            width="stretch"
        )

    else:

        st.warning(
            "Preview image could not be generated."
        )


    # ========================================================
    # METADATA
    # ========================================================

    with st.expander(
        "📊 Image Metadata and Band Statistics"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Image Metadata")

            st.json(
                analysis_result.get(
                    "metadata",
                    {}
                )
            )

        with col2:

            st.subheader("Band Statistics")

            st.json(
                analysis_result.get(
                    "statistics",
                    {}
                )
            )


    # ========================================================
    # 4. ASK SATQUERY AI
    # ========================================================

    st.header("4. Ask SatQuery AI")

    query = st.text_input(
        "Enter your question",
        placeholder="Example: Is there water?"
    )


    analyze_button = st.button(
        "🚀 Analyze Image",
        type="primary"
    )


    # ========================================================
    # RUN COMPLETE AGENT
    # ========================================================

    if analyze_button:

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "🤖 SatQuery AI is analyzing the image..."
                ):

                    result = execute_query(
                        query,
                        [upload_path],
                        [modality]
                    )

                st.session_state.query_result = result

            except Exception as error:

                st.error(
                    f"SatQuery AI execution failed: {error}"
                )

                st.exception(error)


    # ========================================================
    # 5. RESULT
    # ========================================================

    result = st.session_state.query_result

    if result is not None:

        st.divider()

        st.header("5. SatQuery AI Result")

        if result.get("success"):

            specialist_result = result.get(
                "result",
                {}
            )

            answer = specialist_result.get(
                "answer",
                "No answer returned."
            )

            confidence = specialist_result.get(
                "confidence",
                0.0
            )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.success(
                f"Answer: {answer}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.0%}"
            )


            # ------------------------------------------------
            # EXECUTION PLAN
            # ------------------------------------------------

            st.subheader(
                "🤖 Agent Decision"
            )

            plan = result.get(
                "plan",
                {}
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Selected Task",
                    str(
                        plan.get(
                            "task",
                            "unknown"
                        )
                    ).upper()
                )

            with col2:

                st.metric(
                    "Specialist",
                    plan.get(
                        "specialist",
                        "Unknown"
                    )
                )

            with col3:

                st.metric(
                    "Routing Confidence",
                    f"{plan.get('confidence', 0):.0%}"
                )


            st.info(
                f"""
                **Routing reason:**  
                {plan.get('reason', 'Not available.')}

                **Validation:**  
                {plan.get('validation_message', 'Not available.')}
                """
            )


            # ------------------------------------------------
            # EVIDENCE
            # ------------------------------------------------

            st.subheader(
                "🔎 Evidence"
            )

            evidence = specialist_result.get(
                "evidence",
                []
            )

            if evidence:

                for item in evidence:

                    st.write(
                        f"• {item}"
                    )

            else:

                st.info(
                    "No evidence records were returned."
                )


            # ------------------------------------------------
            # OUTPUT DETAILS
            # ------------------------------------------------

            with st.expander(
                "🔧 Specialist Output"
            ):

                st.json(
                    specialist_result.get(
                        "outputs",
                        {}
                    )
                )


            # ------------------------------------------------
            # COMPLETE EXECUTION TRACE
            # ------------------------------------------------

            with st.expander(
                "🧾 Complete Execution Trace"
            ):

                st.json(result)

        else:

            st.error(
                "SatQuery AI could not complete the request."
            )

            st.json(result)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SatQuery AI • Agentic Remote-Sensing Vision-Language Assistant"
)