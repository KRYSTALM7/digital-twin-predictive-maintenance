import os
import csv
from typing import List, Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


PROCESSED_CSV_PATH = os.path.join("data", "processed", "spindle_data_with_anomalies.csv")
MODEL_OUTPUT_PATH = os.path.join("outputs", "lstm_model.h5")
SCALER_OUTPUT_PATH = os.path.join("outputs", "lstm_scaler.npz")


FEATURES = ["temperature", "vibration", "current"]
WINDOW_SIZE = 20


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_features_from_csv(
    path: str,
    feature_names: List[str],
    split_filter: str = None,
) -> np.ndarray:
    """Load feature columns from the processed CSV.

    If split_filter is given (e.g. "train"), only rows whose 'split' column
    matches are returned. This is what keeps training/threshold-fitting
    restricted to clean data — the processed CSV only ever injects
    anomalies into split == 'test' rows (see data_preprocessor.py).
    """
    data: List[List[float]] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split_filter is not None and row.get("split") != split_filter:
                continue
            try:
                data.append([float(row[name]) for name in feature_names])
            except Exception:
                # Skip rows with invalid/missing values
                continue
    return np.asarray(data, dtype=np.float32)


def _make_supervised(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[np.ndarray] = []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _fit_scaler(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def _apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _build_model(input_window: int, num_features: int) -> keras.Model:
    inputs = keras.Input(shape=(input_window, num_features))
    x = layers.LSTM(64, return_sequences=False)(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(num_features, activation="linear")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def train_lstm(
    processed_csv_path: str = PROCESSED_CSV_PATH,
    model_output_path: str = MODEL_OUTPUT_PATH,
    scaler_output_path: str = SCALER_OUTPUT_PATH,
    window_size: int = WINDOW_SIZE,
    epochs: int = 20,
) -> str:
    # 1. Load ONLY the clean training split (split == "train"). This is the
    # key fix: previously this loaded the entire processed CSV, including
    # rows with injected anomalies, so the model (and later, thresholds
    # derived from its errors) had already "seen" anomalous patterns before
    # evaluation. Now training data is guaranteed anomaly-free.
    series = _load_features_from_csv(processed_csv_path, FEATURES, split_filter="train")
    if len(series) <= window_size + 1:
        raise ValueError(
            "Not enough clean training data to create supervised windows. "
            "Check that data_preprocessor.py ran and produced a 'split' column."
        )

    # 2. Create supervised windows from clean data only
    X, y = _make_supervised(series, window=window_size)

    # 3. Internal validation split (sequential 80/20) — this is just for
    # early-stopping/monitoring during training, NOT for anomaly
    # evaluation. Real anomaly evaluation happens later on split == "test"
    # rows in detection_and_metrics.py.
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    # 4. Scale using training stats only
    mean, std = _fit_scaler(X_train)
    X_train_s = _apply_scaler(X_train, mean, std)
    X_val_s = _apply_scaler(X_val, mean, std)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)
    y_std[y_std == 0.0] = 1.0
    y_train_s = (y_train - y_mean) / y_std
    y_val_s = (y_val - y_mean) / y_std

    # 5. Build model
    model = _build_model(input_window=window_size, num_features=len(FEATURES))

    # 6. Train
    model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=epochs,
        batch_size=64,
        verbose=1,
        shuffle=False,
    )

    # 7. Save model and scaler
    out_dir = os.path.dirname(model_output_path)
    if out_dir:
        _ensure_dir(out_dir)
    model.save(model_output_path)

    np.savez(
        scaler_output_path,
        X_mean=mean,
        X_std=std,
        y_mean=y_mean,
        y_std=y_std,
        features=np.asarray(FEATURES),
        window=np.asarray([window_size], dtype=np.int32),
    )
    return model_output_path


def predict_lstm(
    input_sequence: np.ndarray,
    model_path: str = MODEL_OUTPUT_PATH,
    scaler_path: str = SCALER_OUTPUT_PATH,
) -> np.ndarray:
    """Predict next timestep for a single input sequence.

    input_sequence: shape (WINDOW_SIZE, 3) with features [temperature, vibration, current]
    Returns: np.ndarray of shape (3,) in original scale
    """
    if input_sequence.ndim != 2 or input_sequence.shape[0] != WINDOW_SIZE or input_sequence.shape[1] != len(FEATURES):
        raise ValueError(f"input_sequence must be shape ({WINDOW_SIZE}, {len(FEATURES)})")

    model = keras.models.load_model(model_path, compile=False)
    data = np.load(scaler_path, allow_pickle=True)
    X_mean = data["X_mean"]
    X_std = data["X_std"]
    y_mean = data["y_mean"]
    y_std = data["y_std"]

    X_input = input_sequence.astype(np.float32)[None, ...]
    X_input_s = (X_input - X_mean) / X_std
    y_pred_s = model.predict(X_input_s, verbose=0)[0]
    y_pred = y_pred_s * y_std + y_mean
    return y_pred.astype(np.float32)


if __name__ == "__main__":
    path = train_lstm()
    print(f"LSTM training complete. Model saved to: {path}")