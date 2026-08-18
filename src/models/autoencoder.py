import os
import csv
from typing import List, Tuple, Union

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


PROCESSED_CSV_PATH = os.path.join("data", "processed", "spindle_data_with_anomalies.csv")
MODEL_OUTPUT_PATH = os.path.join("outputs", "autoencoder_model.h5")
META_OUTPUT_PATH = os.path.join("outputs", "autoencoder_meta.npz")

FEATURES = ["temperature", "vibration", "current"]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_features_from_csv(
    path: str,
    feature_names: List[str],
    split_filter: str = None,
) -> np.ndarray:
    """Load feature columns, optionally filtered by the 'split' column
    (e.g. split_filter="train" restricts to clean, anomaly-free rows)."""
    data: List[List[float]] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_filter is not None and row.get("split") != split_filter:
                continue
            try:
                data.append([float(row[name]) for name in feature_names])
            except Exception:
                continue
    return np.asarray(data, dtype=np.float32)


def _standardize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0.0] = 1.0
    standardized = (data - mean) / std
    return standardized, mean, std


def _build_autoencoder(input_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.Dense(8, activation="relu")(x)
    bottleneck = layers.Dense(2, activation="relu", name="bottleneck")(x)
    x = layers.Dense(8, activation="relu")(bottleneck)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_autoencoder(
    processed_csv_path: str = PROCESSED_CSV_PATH,
    model_output_path: str = MODEL_OUTPUT_PATH,
    meta_output_path: str = META_OUTPUT_PATH,
    epochs: int = 50,
    batch_size: int = 64,
) -> str:
    # Train ONLY on the clean training split. Previously this loaded the
    # entire processed CSV (anomalies included), so the reconstruction-error
    # threshold below was fit on data containing the very anomalies it was
    # meant to detect — inflating the threshold and hurting recall/precision
    # in a way that doesn't reflect real-world deployment (where you train
    # on known-normal operation, not on future faults).
    data = _load_features_from_csv(processed_csv_path, FEATURES, split_filter="train")
    if data.size == 0:
        raise ValueError(
            "No clean training data found. Check that data_preprocessor.py "
            "ran and produced a 'split' column with 'train' rows."
        )

    standardized, mean, std = _standardize(data)

    model = _build_autoencoder(input_dim=standardized.shape[1])

    model.fit(
        standardized,
        standardized,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        validation_split=0.1,
    )

    # Threshold is now fit purely on clean-data reconstruction error —
    # this represents "how wrong the model is on normal operation", which
    # is the correct baseline for flagging anomalies on unseen (test) data.
    reconstructions = model.predict(standardized, verbose=0)
    errors = np.mean(np.square(standardized - reconstructions), axis=1)
    threshold = float(errors.mean() + 3 * errors.std())

    out_dir = os.path.dirname(model_output_path)
    if out_dir:
        _ensure_dir(out_dir)
    model.save(model_output_path)

    np.savez(
        meta_output_path,
        mean=mean,
        std=std,
        threshold=np.asarray([threshold], dtype=np.float32),
        features=np.asarray(FEATURES),
    )

    return model_output_path


def _prepare_input(data: Union[np.ndarray, List[List[float]], List[float]]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != len(FEATURES):
            raise ValueError(f"1D input must have length {len(FEATURES)}")
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != len(FEATURES):
        raise ValueError(f"Input data must be shape (n, {len(FEATURES)})")
    return arr


def detect_anomalies(
    data: Union[np.ndarray, List[List[float]], List[float]],
    model_path: str = MODEL_OUTPUT_PATH,
    meta_path: str = META_OUTPUT_PATH,
) -> dict:
    """Detect anomalies using trained autoencoder.

    Returns dict with keys: errors (np.ndarray), threshold (float), mask (np.ndarray bool)
    """
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Model or metadata not found. Train the autoencoder first.")

    model = keras.models.load_model(model_path, compile=False)
    meta = np.load(meta_path, allow_pickle=True)
    mean = meta["mean"]
    std = meta["std"]
    threshold = float(meta["threshold"][0])

    arr = _prepare_input(data)
    standardized = (arr - mean) / std
    reconstructions = model.predict(standardized, verbose=0)
    errors = np.mean(np.square(standardized - reconstructions), axis=1)
    mask = errors > threshold

    return {"errors": errors, "threshold": threshold, "mask": mask}


if __name__ == "__main__":
    path = train_autoencoder()
    print(f"Autoencoder training complete. Model saved to: {path}")