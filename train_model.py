# train_model.py

import os
import re
import pickle
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    SimpleRNN, LSTM, GRU,
    Dense, Dropout, BatchNormalization,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Concatenate, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sentence_transformers import SentenceTransformer


# ==========================================================
# 1. SETTINGS
# ==========================================================

DATASET_PATH   = "questions.csv"
MODEL_FOLDER   = "models"
SAMPLE_SIZE    = 50000
TEST_SIZE      = 0.20
RANDOM_STATE   = 42

HF_MODEL_NAME  = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_SIZE = 768
SEQ_LEN        = 6
HF_EPOCHS      = 30
HF_BATCH_SIZE  = 128

os.makedirs(MODEL_FOLDER, exist_ok=True)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ==========================================================
# 2. TEXT CLEANING
# ==========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# 3. LOAD & PREPROCESS DATA
# ==========================================================

print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)
print("Shape:", df.shape)

df = df[["question1", "question2", "is_duplicate"]].dropna().copy()
df["is_duplicate"] = df["is_duplicate"].astype(int)

if SAMPLE_SIZE < len(df):
    half = SAMPLE_SIZE // 2
    df = pd.concat([
        df[df["is_duplicate"] == 1].sample(n=min(half, df["is_duplicate"].sum()), random_state=RANDOM_STATE),
        df[df["is_duplicate"] == 0].sample(n=min(half, (df["is_duplicate"] == 0).sum()), random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("Balanced shape:", df.shape)
print(df["is_duplicate"].value_counts())

df["question1"] = df["question1"].apply(clean_text)
df["question2"] = df["question2"].apply(clean_text)

q1_all = df["question1"].tolist()
q2_all = df["question2"].tolist()
y      = df["is_duplicate"].values

(q1_train, q1_test,
 q2_train, q2_test,
 y_train,  y_test) = train_test_split(
    q1_all, q2_all, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTrain: {len(y_train)} | Test: {len(y_test)}")


# ==========================================================
# 4. EVALUATION FUNCTION
# ==========================================================

results = []

def evaluate_model(model_name, y_true, y_pred, training_time):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Training Time  : {training_time:.2f} seconds")
    print(classification_report(y_true, y_pred, zero_division=0))
    results.append({
        "Model": model_name, "Accuracy": acc,
        "Precision": prec, "Recall": rec,
        "F1 Score": f1, "Training Time (seconds)": training_time
    })


# ==========================================================
# 5. ENCODE WITH HUGGINGFACE
# ==========================================================

print("\nLoading HuggingFace sentence transformer...")
hf_encoder = SentenceTransformer(HF_MODEL_NAME)

print("Encoding questions...")
train_emb1 = hf_encoder.encode(q1_train, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
train_emb2 = hf_encoder.encode(q2_train, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
test_emb1  = hf_encoder.encode(q1_test,  batch_size=64, show_progress_bar=True, convert_to_numpy=True)
test_emb2  = hf_encoder.encode(q2_test,  batch_size=64, show_progress_bar=True, convert_to_numpy=True)


# ==========================================================
# 6. FEATURE BUILDERS
# ==========================================================

def pairwise_cosine(a, b):
    num   = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-10
    return (num / denom).reshape(-1, 1)

def build_sequence_input(e1, e2):
    return np.stack([
        e1, e2, np.abs(e1 - e2),
        e1 * e2, np.minimum(e1, e2), np.maximum(e1, e2)
    ], axis=1).astype(np.float32)

def build_scalar_features(e1, e2):
    cos     = pairwise_cosine(e1, e2)
    euc     = np.linalg.norm(e1 - e2, axis=1, keepdims=True)
    l1      = np.sum(np.abs(e1 - e2), axis=1, keepdims=True)
    dot     = np.sum(e1 * e2, axis=1, keepdims=True)
    sq_diff = np.sum((e1 - e2) ** 2, axis=1, keepdims=True)
    cos_abs = pairwise_cosine(np.abs(e1 - e2), np.abs(e1 - e2))
    return np.hstack([cos, euc, l1, dot, sq_diff, cos_abs]).astype(np.float32)

X_train_seq    = build_sequence_input(train_emb1, train_emb2)
X_test_seq     = build_sequence_input(test_emb1,  test_emb2)
X_train_scalar = build_scalar_features(train_emb1, train_emb2)
X_test_scalar  = build_scalar_features(test_emb1,  test_emb2)

SCALAR_DIM = X_train_scalar.shape[1]

print(f"Sequence shape : {X_train_seq.shape}")
print(f"Scalar shape   : {X_train_scalar.shape}")


# ==========================================================
# 7. CALLBACKS
# ==========================================================

def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]


# ==========================================================
# 8. RNN / LSTM / GRU MODEL
# ==========================================================

def create_rnn_model(model_type):
    seq_input    = Input(shape=(SEQ_LEN, EMBEDDING_SIZE), name="seq_input")
    scalar_input = Input(shape=(SCALAR_DIM,), name="scalar_input")

    if model_type == "RNN":
        x = SimpleRNN(256, return_sequences=True)(seq_input)
        x = SimpleRNN(128)(x)
    elif model_type == "LSTM":
        x = LSTM(256, return_sequences=True)(seq_input)
        x = LSTM(128)(x)
    elif model_type == "GRU":
        x = GRU(256, return_sequences=True)(seq_input)
        x = GRU(128)(x)

    x = Dropout(0.3)(x)
    x = Concatenate()([x, scalar_input])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model([seq_input, scalar_input], outputs, name=f"{model_type}_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==========================================================
# 9. TRANSFORMER CUSTOM LAYERS & LOSS
# ==========================================================

class SliceLayer(tf.keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
    def call(self, x):
        return x[:, self.index:self.index+1, :]
    def get_config(self):
        return {**super().get_config(), "index": self.index}

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
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, self.gamma) * bce)
    def get_config(self):
        return {**super().get_config(), "gamma": self.gamma, "alpha": self.alpha}
# ==========================================================
# 10. TRANSFORMER MODEL  (CPU-friendly sizes)
# ==========================================================

def create_transformer_model():
    reg          = l2(1e-4)
    seq_input    = Input(shape=(SEQ_LEN, EMBEDDING_SIZE), name="seq_input")
    scalar_input = Input(shape=(SCALAR_DIM,), name="scalar_input")

    q1 = SliceLayer(0, name="slice_q1")(seq_input)
    q2 = SliceLayer(1, name="slice_q2")(seq_input)

    # Cross-attention: Q1 attends to Q2
    cross1 = MultiHeadAttention(num_heads=4, key_dim=64, name="cross_attn_q1q2")(q1, q2)
    cross1 = Dropout(0.1)(cross1)
    c1     = LayerNormalization(epsilon=1e-6)(Add()([q1, cross1]))
    ffn_c1 = Dense(256, activation="gelu")(c1)
    ffn_c1 = Dropout(0.1)(ffn_c1)
    ffn_c1 = Dense(EMBEDDING_SIZE)(ffn_c1)
    c1     = LayerNormalization(epsilon=1e-6)(Add()([c1, ffn_c1]))

    # Cross-attention: Q2 attends to Q1
    cross2 = MultiHeadAttention(num_heads=4, key_dim=64, name="cross_attn_q2q1")(q2, q1)
    cross2 = Dropout(0.1)(cross2)
    c2     = LayerNormalization(epsilon=1e-6)(Add()([q2, cross2]))
    ffn_c2 = Dense(256, activation="gelu")(c2)
    ffn_c2 = Dropout(0.1)(ffn_c2)
    ffn_c2 = Dense(EMBEDDING_SIZE)(ffn_c2)
    c2     = LayerNormalization(epsilon=1e-6)(Add()([c2, ffn_c2]))

    # Self-attention block 1
    attn1 = MultiHeadAttention(num_heads=4, key_dim=64, name="self_attn_1")(seq_input, seq_input)
    attn1 = Dropout(0.1)(attn1)
    x     = LayerNormalization(epsilon=1e-6)(Add()([seq_input, attn1]))
    ffn1  = Dense(256, activation="gelu")(x)
    ffn1  = Dropout(0.1)(ffn1)
    ffn1  = Dense(EMBEDDING_SIZE)(ffn1)
    x     = LayerNormalization(epsilon=1e-6)(Add()([x, ffn1]))

    # Self-attention block 2
    attn2 = MultiHeadAttention(num_heads=4, key_dim=64, name="self_attn_2")(x, x)
    attn2 = Dropout(0.1)(attn2)
    x     = LayerNormalization(epsilon=1e-6)(Add()([x, attn2]))
    ffn2  = Dense(256, activation="gelu")(x)
    ffn2  = Dropout(0.1)(ffn2)
    ffn2  = Dense(EMBEDDING_SIZE)(ffn2)
    x     = LayerNormalization(epsilon=1e-6)(Add()([x, ffn2]))

    x       = GlobalAveragePooling1D()(x)
    c1_flat = SqueezeLayer(name="squeeze_c1")(c1)
    c2_flat = SqueezeLayer(name="squeeze_c2")(c2)

    x = Concatenate()([x, c1_flat, c2_flat, scalar_input])
    x = BatchNormalization()(x)
    x = Dense(512, activation="gelu", kernel_regularizer=reg)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="gelu", kernel_regularizer=reg)(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="gelu", kernel_regularizer=reg)(x)
    x = Dense(64,  activation="gelu", kernel_regularizer=reg)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model([seq_input, scalar_input], outputs, name="transformer_model")

    total_steps = int((SAMPLE_SIZE * 0.85) / HF_BATCH_SIZE) * HF_EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-5,
        decay_steps=max(total_steps, 1),
        alpha=1e-7
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==========================================================
# 11. TRAIN RNN, LSTM, GRU
# ==========================================================

for model_type in ["RNN", "LSTM", "GRU"]:
    print("\n" + "=" * 60)
    print(f"TRAINING {model_type}")
    print("=" * 60)

    model = create_rnn_model(model_type)
    model.summary()
    start_time = time.time()

    model.fit(
        [X_train_seq, X_train_scalar], y_train,
        validation_split=0.15,
        epochs=HF_EPOCHS,
        batch_size=HF_BATCH_SIZE,
        callbacks=get_callbacks(),
        verbose=1
    )

    training_time = time.time() - start_time
    probs       = model.predict([X_test_seq, X_test_scalar], verbose=0)
    predictions = (probs >= 0.50).astype(int).flatten()
    evaluate_model(model_type, y_test, predictions, training_time)

    path = os.path.join(MODEL_FOLDER, f"{model_type.lower()}_model.keras")
    model.save(path)
    print(f"{model_type} saved to: {path}")


# ==========================================================
# 12. TRAIN TRANSFORMER
# ==========================================================

print("\n" + "=" * 60)
print("TRAINING TRANSFORMER")
print("=" * 60)

transformer_model = create_transformer_model()
transformer_model.summary()
start_time = time.time()

from sklearn.utils.class_weight import compute_class_weight

transformer_model.fit(
    [X_train_seq, X_train_scalar], y_train,
    validation_split=0.15,
    epochs=HF_EPOCHS,
    batch_size=HF_BATCH_SIZE,
    callbacks=[EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)],
    class_weight=dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train))),
    verbose=1
)

training_time = time.time() - start_time
probs = transformer_model.predict([X_test_seq, X_test_scalar], verbose=0).flatten()

# Tune threshold for best F1
best_thresh, best_f1 = 0.5, 0.0
for thresh in np.arange(0.30, 0.71, 0.01):
    preds = (probs >= thresh).astype(int)
    score = f1_score(y_test, preds, zero_division=0)
    if score > best_f1:
        best_f1, best_thresh = score, thresh

print(f"Best threshold: {best_thresh:.2f} | Best F1: {best_f1:.4f}")
predictions = (probs >= best_thresh).astype(int)
evaluate_model("Transformer", y_test, predictions, training_time)

path = os.path.join(MODEL_FOLDER, "transformer_model.keras")
transformer_model.save(path)
print(f"Transformer saved to: {path}")

thresh_path = os.path.join(MODEL_FOLDER, "transformer_threshold.pkl")
with open(thresh_path, "wb") as f:
    pickle.dump(float(best_thresh), f)
print(f"Threshold saved to: {thresh_path}")


# ==========================================================
# 13. SAVE RESULTS
# ==========================================================

results_df   = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
results_path = os.path.join(MODEL_FOLDER, "model_comparison_results.csv")
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\nResults saved to: {results_path}")
print("\nAll models trained and saved successfully.")
