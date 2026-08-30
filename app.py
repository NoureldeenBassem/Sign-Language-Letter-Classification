"""
Sign Language Letter Classification - Streamlit app
Author: Eng. Noureldin Bassem Mohamed

Upload a photo of an American Sign Language hand sign and the model says which
of the 24 letters it is (the alphabet without J and Z, both of which need
movement and cannot be shown in a still photo).

The model is the CNN trained from scratch in sign-language-cnn.ipynb. It is only
loaded here, never retrained. An uploaded photo goes through exactly the same
preprocessing the training images went through: grayscale, resized to 28x28,
scaled to [0, 1].
"""

import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "asl_cnn.keras")
LABELS_PATH = os.path.join(BASE_DIR, "artifacts", "class_names.json")

# below this the model is not really sure, a real product would ask the user to
# retake the photo instead of showing the letter as if it were settled
LOW_CONFIDENCE = 0.60


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        meta = json.load(f)
    return model, meta["class_names"], int(meta["img_size"])


def preprocess_photo(image, img_size, invert=False):
    """Turn any uploaded photo into the 1x28x28x1 array the model expects."""
    img = image.convert("L").resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    if invert:
        arr = 1.0 - arr
    return arr.reshape(1, img_size, img_size, 1)


st.set_page_config(page_title="ASL Letter Classification", page_icon="🤟")

st.title("🤟 Sign Language Letter Classification")
st.write(
    "Upload a photo of an ASL hand sign and the model predicts the letter. "
    "The model is a CNN trained from scratch on the Sign Language MNIST dataset, "
    "24 classes: the alphabet without J and Z."
)

model, class_names, img_size = load_model()

with st.sidebar:
    st.header("Options")
    invert = st.checkbox(
        "Invert colours",
        value=False,
        help=(
            "The training images are a light hand on a darker background. If your "
            "photo is the other way round, turn this on."
        ),
    )
    st.caption(
        "The training images are tight crops of a hand on a plain background, so "
        "crop your photo close to the hand for the best result."
    )

uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Waiting for a photo. The letters the model knows: " + ", ".join(class_names) + ".")
else:
    image = Image.open(uploaded)
    batch = preprocess_photo(image, img_size, invert=invert)
    probabilities = model.predict(batch, verbose=0)[0]

    top = probabilities.argsort()[::-1][:3]
    letter = class_names[top[0]]
    confidence = float(probabilities[top[0]])

    left, right = st.columns(2)
    with left:
        st.image(image, caption="Your photo", use_container_width=True)
    with right:
        st.image(
            batch[0].squeeze(),
            caption=f"What the model sees ({img_size}x{img_size} grayscale)",
            use_container_width=True,
            clamp=True,
        )

    st.subheader(f"Prediction: {letter}")
    st.write(f"Confidence: {confidence:.1%}")

    if confidence < LOW_CONFIDENCE:
        st.warning(
            "The model is not confident about this one. Try cropping closer to the "
            "hand, or toggling 'Invert colours' in the sidebar."
        )

    st.write("Top 3 letters")
    st.bar_chart({"probability": {class_names[i]: float(probabilities[i]) for i in top}})

    with st.expander("All 24 probabilities"):
        st.dataframe(
            {
                "letter": class_names,
                "probability": [round(float(p), 4) for p in probabilities],
            },
            use_container_width=True,
        )
