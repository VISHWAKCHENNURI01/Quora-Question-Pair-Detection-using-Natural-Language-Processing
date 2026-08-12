import os
import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Quora Question Pair Detection",
    page_icon="🔍",
    layout="centered"
)


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_FOLDER = "models"

MODEL_FILES = {
    "RNN": "rnn_model.keras",
    "LSTM": "lstm_model.keras",
    "GRU": "gru_model.keras",
    "Transformer": "transformer_model.keras"
}

# Starting threshold for the dataset-free MiniLM model.
# Lowering it from 0.75 helps detect paraphrases.
MINILM_THRESHOLD = 0.65

# Your RNN/LSTM/GRU training code used 0.50.
DEEP_MODEL_THRESHOLD = 0.50


# ==========================================================
# CUSTOM LAYERS / LOSS
# Needed for the Transformer trained with your train_model.py
# ==========================================================

@tf.keras.utils.register_keras_serializable()
class SliceLayer(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, x):
        return x[:, self.index:self.index + 1, :]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config


@tf.keras.utils.register_keras_serializable()
class SqueezeLayer(tf.keras.layers.Layer):

    def call(self, x):
        return tf.squeeze(x, axis=1)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        bce = (
            -y_true * tf.math.log(y_pred)
            - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        p_t = (
            y_true * y_pred
            + (1.0 - y_true) * (1.0 - y_pred)
        )

        alpha_t = (
            y_true * self.alpha
            + (1.0 - y_true) * (1.0 - self.alpha)
        )

        return tf.reduce_mean(
            alpha_t *
            tf.pow(1.0 - p_t, self.gamma) *
            bce
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config


# ==========================================================
# LOAD KERAS MODEL
# ==========================================================

@st.cache_resource
def load_keras_model(model_name):

    model_path = os.path.join(
        MODEL_FOLDER,
        MODEL_FILES[model_name]
    )

    if not os.path.isfile(model_path):
        return None

    custom_objects = {
        "SliceLayer": SliceLayer,
        "SqueezeLayer": SqueezeLayer,
        "FocalLoss": FocalLoss
    }

    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not load {model_name} model: {e}"
        )


# ==========================================================
# LOAD SENTENCE TRANSFORMER
# ==========================================================

@st.cache_resource
def load_encoder(encoder_name):

    if encoder_name == "MiniLM":
        return SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    if encoder_name == "MPNet":
        return SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2"
        )

    raise ValueError(
        f"Unknown encoder: {encoder_name}"
    )


# ==========================================================
# TEXT CLEANING
# Must match the training preprocessing closely.
# ==========================================================

def clean_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_for_exact_match(text):

    return " ".join(
        clean_text(text).split()
    )


# ==========================================================
# COSINE SIMILARITY
# ==========================================================

def pairwise_cosine(a, b):

    numerator = np.sum(
        a * b,
        axis=1
    )

    denominator = (
        np.linalg.norm(a, axis=1)
        *
        np.linalg.norm(b, axis=1)
        +
        1e-10
    )

    return (
        numerator / denominator
    ).reshape(-1, 1)


# ==========================================================
# SEQUENCE FEATURES
# Same structure as train_model.py
# ==========================================================

def build_sequence_features(e1, e2):

    return np.stack(
        [
            e1,
            e2,
            np.abs(e1 - e2),
            e1 * e2,
            np.minimum(e1, e2),
            np.maximum(e1, e2)
        ],
        axis=1
    ).astype(np.float32)


# ==========================================================
# SCALAR FEATURES
# Supports both:
#   4 features -> fast training version
#   6 features -> original training version
# ==========================================================

def build_scalar_features(
    e1,
    e2,
    scalar_dimension
):

    cosine = pairwise_cosine(e1, e2)

    euclidean = np.linalg.norm(
        e1 - e2,
        axis=1,
        keepdims=True
    )

    manhattan = np.sum(
        np.abs(e1 - e2),
        axis=1,
        keepdims=True
    )

    dot_product = np.sum(
        e1 * e2,
        axis=1,
        keepdims=True
    )

    if scalar_dimension == 4:

        return np.hstack(
            [
                cosine,
                euclidean,
                manhattan,
                dot_product
            ]
        ).astype(np.float32)

    if scalar_dimension == 6:

        squared_difference = np.sum(
            (e1 - e2) ** 2,
            axis=1,
            keepdims=True
        )

        # Keep this exactly compatible with the original
        # train_model.py feature construction.
        abs_difference = np.abs(e1 - e2)

        # Note: cosine(abs_diff, abs_diff) is mathematically 1.
        cosine_abs = pairwise_cosine(
            abs_difference,
            abs_difference
        )

        return np.hstack(
            [
                cosine,
                euclidean,
                manhattan,
                dot_product,
                squared_difference,
                cosine_abs
            ]
        ).astype(np.float32)

    raise ValueError(
        f"Unsupported scalar dimension: {scalar_dimension}"
    )


# ==========================================================
# MODEL DIMENSIONS
# ==========================================================

def get_model_dimensions(model):

    sequence_shape = model.inputs[0].shape
    scalar_shape = model.inputs[1].shape

    seq_len = int(sequence_shape[1])
    embedding_size = int(sequence_shape[2])
    scalar_dimension = int(scalar_shape[1])

    return (
        seq_len,
        embedding_size,
        scalar_dimension
    )


# ==========================================================
# SELECT ENCODER BASED ON TRAINED MODEL
# ==========================================================

def get_encoder_for_model(model):

    (
        seq_len,
        embedding_size,
        scalar_dimension
    ) = get_model_dimensions(model)

    if embedding_size == 384:
        encoder_name = "MiniLM"

    elif embedding_size == 768:
        encoder_name = "MPNet"

    else:
        raise ValueError(
            f"Unsupported embedding size: {embedding_size}. "
            f"Expected 384 or 768."
        )

    encoder = load_encoder(
        encoder_name
    )

    return (
        encoder,
        encoder_name,
        seq_len,
        embedding_size,
        scalar_dimension
    )


# ==========================================================
# DEEP MODEL PREDICTION
# ==========================================================

def predict_deep_model(
    model,
    question1,
    question2
):

    question1 = clean_text(question1)
    question2 = clean_text(question2)

    (
        encoder,
        encoder_name,
        seq_len,
        embedding_size,
        scalar_dimension
    ) = get_encoder_for_model(model)

    embeddings = encoder.encode(
        [
            question1,
            question2
        ],
        batch_size=2,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    e1 = embeddings[0].reshape(
        1,
        embedding_size
    )

    e2 = embeddings[1].reshape(
        1,
        embedding_size
    )

    X_seq = build_sequence_features(
        e1,
        e2
    )

    X_scalar = build_scalar_features(
        e1,
        e2,
        scalar_dimension
    )

    probability = model.predict(
        [
            X_seq,
            X_scalar
        ],
        verbose=0
    )

    probability = float(
        np.asarray(probability).reshape(-1)[0]
    )

    probability = float(
        np.clip(probability, 0.0, 1.0)
    )

    return {
        "probability": probability,
        "encoder_name": encoder_name,
        "embedding_size": embedding_size,
        "scalar_dimension": scalar_dimension,
        "seq_len": seq_len
    }


# ==========================================================
# MINILM PREDICTION
# ==========================================================

def predict_minilm(
    question1,
    question2
):

    encoder = load_encoder(
        "MiniLM"
    )

    embeddings = encoder.encode(
        [
            clean_text(question1),
            clean_text(question2)
        ],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(
        np.clip(similarity, 0.0, 1.0)
    )


# ==========================================================
# MODEL STATUS
# ==========================================================

def get_model_status():

    status = {}

    for model_name, filename in MODEL_FILES.items():

        path = os.path.join(
            MODEL_FOLDER,
            filename
        )

        status[model_name] = os.path.isfile(path)

    return status


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "🔍 Quora Question Pair Detection"
)

st.write(
    "Detect whether two questions have the same "
    "semantic meaning using NLP and Deep Learning."
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header(
        "About the Project"
    )

    st.write(
        """
        **Prediction**

        1 = Duplicate / Same Meaning

        0 = Non-Duplicate / Different Meaning

        **Available Models**

        • Sentence Transformer (MiniLM)
        • RNN
        • LSTM
        • GRU
        • Transformer
        """
    )

    st.divider()

    st.subheader(
        "Model Status"
    )

    status = get_model_status()

    for name, available in status.items():

        if available:
            st.success(
                f"{name}: Loaded"
            )
        else:
            st.error(
                f"{name}: Model file missing"
            )

    st.divider()

    st.subheader(
        "Technologies"
    )

    st.write(
        """
        • Python
        • Streamlit
        • NLP
        • Sentence Transformers
        • TensorFlow
        • RNN
        • LSTM
        • GRU
        • Transformer
        """
    )


# ==========================================================
# MODEL SELECTION
# ==========================================================

st.subheader(
    "🤖 Select Prediction Model"
)

model_option = st.selectbox(
    "Choose a model",
    [
        "Sentence Transformer (MiniLM)",
        "RNN",
        "LSTM",
        "GRU",
        "Transformer"
    ]
)


# ==========================================================
# LOAD SELECTED MODEL
# ==========================================================

selected_model = None

if model_option != "Sentence Transformer (MiniLM)":

    try:

        selected_model = load_keras_model(
            model_option
        )

    except Exception as e:

        st.error(
            f"❌ Error loading {model_option}."
        )

        st.exception(e)

        st.stop()

    if selected_model is None:

        st.error(
            f"❌ {model_option} model file was not found."
        )

        st.info(
            "Place the trained .keras file inside the "
            "'models' folder."
        )

        st.code(
            os.path.join(
                MODEL_FOLDER,
                MODEL_FILES[model_option]
            )
        )

        st.stop()

    (
        seq_len,
        embedding_size,
        scalar_dimension
    ) = get_model_dimensions(
        selected_model
    )

    st.success(
        f"✅ {model_option} model loaded"
    )

    st.caption(
        f"Embedding: {embedding_size}D | "
        f"Sequence length: {seq_len} | "
        f"Scalar features: {scalar_dimension}"
    )

else:

    st.success(
        "✅ Sentence Transformer (MiniLM) selected"
    )

    st.caption(
        "Pre-trained dataset-free semantic model"
    )


# ==========================================================
# QUESTION INPUT
# ==========================================================

st.subheader(
    "📝 Enter Questions"
)

question1 = st.text_area(
    "Question 1",
    placeholder="Example: How can I learn Python?",
    height=100
)

question2 = st.text_area(
    "Question 2",
    placeholder="Example: What is the best way to learn Python?",
    height=100
)


# ==========================================================
# THRESHOLD
# ==========================================================

if model_option == "Sentence Transformer (MiniLM)":

    threshold = st.slider(
        "Semantic Similarity Threshold",
        min_value=0.40,
        max_value=0.95,
        value=MINILM_THRESHOLD,
        step=0.01
    )

else:

    threshold = DEEP_MODEL_THRESHOLD

    st.info(
        f"{model_option} uses a classification threshold "
        f"of {threshold:.2f}."
    )


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

if st.button(
    "🔍 Predict",
    use_container_width=True
):

    # ------------------------------------------------------
    # Validate input
    # ------------------------------------------------------

    if (
        not question1.strip()
        or not question2.strip()
    ):

        st.warning(
            "⚠️ Please enter both questions."
        )

        st.stop()


    # ======================================================
    # EXACT MATCH CHECK
    # ======================================================

    q1_normalized = normalize_for_exact_match(
        question1
    )

    q2_normalized = normalize_for_exact_match(
        question2
    )

    if q1_normalized == q2_normalized:

        prediction = 1
        score = 1.0

        st.subheader(
            "📊 Prediction Result"
        )

        st.success(
            "✅ DUPLICATE / SAME QUESTION"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                "1"
            )

        with col2:
            st.metric(
                "Similarity",
                "100%"
            )

        with col3:
            st.metric(
                "Threshold",
                f"{threshold * 100:.0f}%"
            )

        st.info(
            "The two questions are exactly the same "
            "after basic text normalization."
        )

        st.stop()


    # ======================================================
    # SENTENCE TRANSFORMER
    # ======================================================

    if model_option == "Sentence Transformer (MiniLM)":

        with st.spinner(
            "Calculating semantic similarity..."
        ):

            try:

                similarity = predict_minilm(
                    question1,
                    question2
                )

            except Exception as e:

                st.error(
                    "❌ Sentence Transformer prediction failed."
                )

                st.exception(e)

                st.stop()

        score = similarity

        if score >= threshold:

            prediction = 1
            result = "Duplicate / Same Meaning"

        else:

            prediction = 0
            result = "Non-Duplicate / Different Meaning"


        st.subheader(
            "📊 Prediction Result"
        )

        if prediction == 1:

            st.success(
                "✅ DUPLICATE / SAME MEANING"
            )

        else:

            st.error(
                "❌ NON-DUPLICATE / DIFFERENT MEANING"
            )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                str(prediction)
            )

        with col2:

            st.metric(
                "Similarity",
                f"{score * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Threshold",
                f"{threshold * 100:.0f}%"
            )


        st.progress(
            float(score)
        )


        if prediction == 1:

            st.success(
                f"The questions are predicted as "
                f"**Duplicate (1)** because their "
                f"semantic similarity is "
                f"**{score * 100:.2f}%**."
            )

        else:

            st.info(
                f"The questions are predicted as "
                f"**Non-Duplicate (0)** because their "
                f"semantic similarity is "
                f"**{score * 100:.2f}%**."
            )


    # ======================================================
    # RNN / LSTM / GRU / TRANSFORMER
    # ======================================================

    else:

        with st.spinner(
            f"Running {model_option} prediction..."
        ):

            try:

                output = predict_deep_model(
                    selected_model,
                    question1,
                    question2
                )

            except Exception as e:

                st.error(
                    f"❌ {model_option} prediction failed."
                )

                st.exception(e)

                st.stop()


        probability = output["probability"]

        prediction = (
            1
            if probability >= threshold
            else 0
        )

        if prediction == 1:

            result = "Duplicate / Same Meaning"

        else:

            result = "Non-Duplicate / Different Meaning"


        # ==================================================
        # RESULT
        # ==================================================

        st.subheader(
            "📊 Prediction Result"
        )

        if prediction == 1:

            st.success(
                "✅ DUPLICATE / SAME MEANING"
            )

        else:

            st.error(
                "❌ NON-DUPLICATE / DIFFERENT MEANING"
            )


        # ==================================================
        # METRICS
        # ==================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                str(prediction)
            )

        with col2:

            st.metric(
                "Model Probability",
                f"{probability * 100:.2f}%"
            )

        with col3:

            st.metric(
                "Threshold",
                f"{threshold * 100:.0f}%"
            )


        st.progress(
            float(probability)
        )


        # ==================================================
        # MODEL INFORMATION
        # ==================================================

        st.subheader(
            "🔎 Model Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.write(
                "**Model**"
            )

            st.write(
                model_option
            )

        with col2:

            st.write(
                "**Embedding Model**"
            )

            st.write(
                output["encoder_name"]
            )

        with col3:

            st.write(
                "**Embedding Size**"
            )

            st.write(
                f'{output["embedding_size"]}D'
            )

        st.write(
            f'**Sequence Length:** {output["seq_len"]}'
        )

        st.write(
            f'**Scalar Features:** '
            f'{output["scalar_dimension"]}'
        )


        # ==================================================
        # INTERPRETATION
        # ==================================================

        st.subheader(
            "💡 Interpretation"
        )

        if prediction == 1:

            st.success(
                f"The {model_option} predicts "
                f"**1 (Duplicate)** with a model "
                f"probability of "
                f"**{probability * 100:.2f}%**."
            )

        else:

            st.info(
                f"The {model_option} predicts "
                f"**0 (Non-Duplicate)** with a model "
                f"probability of "
                f"**{probability * 100:.2f}%**."
            )


    # ======================================================
    # QUESTION DISPLAY
    # ======================================================

    st.divider()

    st.subheader(
        "📝 Input Questions"
    )

    st.write(
        "**Question 1:**"
    )

    st.info(
        question1
    )

    st.write(
        "**Question 2:**"
    )

    st.info(
        question2
    )
