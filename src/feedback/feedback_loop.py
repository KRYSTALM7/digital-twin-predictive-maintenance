import os
import csv
from typing import Tuple

import numpy as np
from tensorflow import keras

from src.models.lstm_forecast import (
    train_lstm,
    PROCESSED_CSV_PATH,
    MODEL_OUTPUT_PATH as LSTM_MODEL_PATH,
    SCALER_OUTPUT_PATH as LSTM_SCALER_PATH,
    FEATURES as LSTM_FEATURES,
    WINDOW_SIZE as LSTM_WINDOW_SIZE,
)
from src.models.autoencoder import (
    train_autoencoder,
    detect_anomalies,
    MODEL_OUTPUT_PATH as AE_MODEL_PATH,
    META_OUTPUT_PATH as AE_META_PATH,
)


FEEDBACK_LOG_PATH = os.path.join("outputs", "feedback_log.csv")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_processed_dataset(path: str = PROCESSED_CSV_PATH) -> Tuple[np.ndarray, np.ndarray]:
    timestamps = []
    features = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamps.append(float(row["timestamp"]))
                features.append([float(row[name]) for name in LSTM_FEATURES])
            except Exception:
                continue
    if not features:
        raise ValueError("Processed dataset is empty; run preprocessing first.")
    return np.asarray(timestamps, dtype=np.float32), np.asarray(features, dtype=np.float32)


def _ensure_models_ready() -> None:
    if not (os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH)):
        train_lstm()
    if not (os.path.exists(AE_MODEL_PATH) and os.path.exists(AE_META_PATH)):
        train_autoencoder()


def _load_lstm_artifacts():
    model = keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    scaler = np.load(LSTM_SCALER_PATH, allow_pickle=True)
    X_mean = scaler["X_mean"]
    X_std = scaler["X_std"]
    y_mean = scaler["y_mean"]
    y_std = scaler["y_std"]
    return model, X_mean, X_std, y_mean, y_std


def _predict_next(
    model: keras.Model,
    sequence: np.ndarray,
    X_mean: np.ndarray,
    X_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    seq = sequence.astype(np.float32)[None, ...]
    seq_s = (seq - X_mean) / X_std
    pred_s = model.predict(seq_s, verbose=0)[0]
    pred = pred_s * y_std + y_mean
    return pred.astype(np.float32)


def run_feedback_loop(
    processed_csv_path: str = PROCESSED_CSV_PATH,
    feedback_log_path: str = FEEDBACK_LOG_PATH,
) -> str:
    """Run feedback loop combining LSTM forecasts and autoencoder anomalies."""

    _ensure_models_ready()

    timestamps, features = _load_processed_dataset(processed_csv_path)
    if len(features) <= LSTM_WINDOW_SIZE:
        raise ValueError("Dataset too small for feedback loop.")

    lstm_model, X_mean, X_std, y_mean, y_std = _load_lstm_artifacts()

    # Forecast predictions and errors
    forecast_errors = []
    forecasts = []
    for idx in range(LSTM_WINDOW_SIZE, len(features)):
        seq = features[idx - LSTM_WINDOW_SIZE : idx]
        pred = _predict_next(lstm_model, seq, X_mean, X_std, y_mean, y_std)
        actual = features[idx]
        error = float(np.linalg.norm(actual - pred))
        forecasts.append(pred)
        forecast_errors.append(error)

    forecast_errors_arr = np.asarray(forecast_errors, dtype=np.float32)
    if forecast_errors_arr.size == 0:
        raise ValueError("Unable to compute forecast errors; check dataset size.")
    forecast_threshold = float(forecast_errors_arr.mean() + 3 * forecast_errors_arr.std())

    # Autoencoder anomaly detection on aligned samples
    auto_results = detect_anomalies(features[LSTM_WINDOW_SIZE:])
    auto_errors = auto_results["errors"]
    auto_threshold = float(auto_results["threshold"])
    auto_mask = auto_results["mask"]

    decisions = []
    for i, error in enumerate(forecast_errors_arr):
        forecast_flag = error > forecast_threshold
        auto_flag = bool(auto_mask[i])
        if forecast_flag and auto_flag:
            decision = "Maintenance Required"
        elif forecast_flag or auto_flag:
            decision = "Monitor"
        else:
            decision = "Normal Operation"
        decisions.append(
            {
                "timestamp": timestamps[i + LSTM_WINDOW_SIZE],
                "forecast_error": error,
                "forecast_threshold": forecast_threshold,
                "auto_error": float(auto_errors[i]),
                "auto_threshold": auto_threshold,
                "decision": decision,
            }
        )

    out_dir = os.path.dirname(feedback_log_path)
    if out_dir:
        _ensure_dir(out_dir)

    with open(feedback_log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "forecast_error",
                "forecast_threshold",
                "auto_error",
                "auto_threshold",
                "decision",
            ],
        )
        writer.writeheader()
        for row in decisions:
            writer.writerow(row)

    return feedback_log_path


if __name__ == "__main__":
    path = run_feedback_loop()
    print(f"Feedback loop complete. Log saved to: {path}")


