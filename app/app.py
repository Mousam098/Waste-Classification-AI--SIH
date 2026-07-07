import os
import sys
from PIL import Image
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from predict import predict_waste, load_model


st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }

    .hero {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, #e8f5e9, #e3f2fd);
        text-align: center;
        margin-bottom: 28px;
        border: 1px solid #d8e6dd;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 850;
        color: #12372a;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #3d4f5c;
    }

    .card {
        padding: 24px;
        border-radius: 20px;
        background-color: white;
        border: 1px solid #edf2f7;
        box-shadow: 0px 8px 26px rgba(15, 23, 42, 0.08);
        margin-top: 16px;
    }

    .bio {
        color: #166534;
        font-size: 34px;
        font-weight: 850;
    }

    .nonbio {
        color: #991b1b;
        font-size: 34px;
        font-weight: 850;
    }

    .metric-text {
        font-size: 17px;
        color: #334155;
        line-height: 1.7;
    }

    .footer {
        text-align: center;
        margin-top: 45px;
        color: #64748b;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">♻️ Waste Classification AI</div>
        <div class="hero-subtitle">
            Classify waste images into Biodegradable and Non-Biodegradable categories
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

model, is_trained = load_model()

if is_trained:
    st.success("Custom trained model loaded from models/waste_model.json")
else:
    st.warning("No trained model found. The app is using the built-in default model. You can train your own model using python src/train_model.py")

left, right = st.columns([1, 1])

with left:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    st.markdown(
        """
        <div class="card">
            <p class="metric-text">
                <b>Biodegradable examples:</b> food waste, leaves, fruit peel, vegetable waste, paper, cardboard.
                <br>
                <b>Non-Biodegradable examples:</b> plastic bottle, wrapper, glass, metal can, e-waste.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with right:
    st.subheader("Result")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="card">
                <p class="metric-text">Upload an image to get prediction result.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Waste Image", use_column_width=True)

        uploaded_file.seek(0)

        with st.spinner("Analyzing image..."):
            result = predict_waste(uploaded_file)

        label_class = "bio" if result["class_key"] == "biodegradable" else "nonbio"

        st.markdown(
            f"""
            <div class="card">
                <div class="{label_class}">{result["label"]}</div>
                <p class="metric-text"><b>Confidence:</b> {result["confidence"]}%</p>
                <p class="metric-text"><b>Suggestion:</b> {result["suggestion"]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(result["confidence"]))

        with st.expander("Technical Details"):
            st.write("Biodegradable distance:", result["bio_distance"])
            st.write("Non-biodegradable distance:", result["non_bio_distance"])
            st.write("Training counts:", result["training_counts"])
            st.write("Extracted image features:")
            st.json(result["features"])

st.markdown(
    """
    <div class="footer">
        Built using Streamlit and Pillow. Python 3.13 compatible.
    </div>
    """,
    unsafe_allow_html=True
)
