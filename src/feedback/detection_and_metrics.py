import os
import numpy as np
import pandas as pd
from typing import Optional, Dict

from src.models.lstm_forecast import (
    train_lstm,
    PROCESSED_CSV_PATH as LSTM_PROCESSED_CSV,
    MODEL_OUTPUT_PATH as LSTM_MODEL_PATH,
    SCALER_OUTPUT_PATH as LSTM_SCALER_PATH,
    WINDOW_SIZE as LSTM_WINDOW_SIZE,
    FEATURES as LSTM_FEATURES,
)
from src.models.autoencoder import (
    train_autoencoder,
    detect_anomalies,
    MODEL_OUTPUT_PATH as AE_MODEL_PATH,
    META_OUTPUT_PATH as AE_META_PATH,
)
from tensorflow import keras

OUT_RESULTS_CSV = os.path.join("outputs", "detection_results.csv")

# How many standard deviations above the CLEAN (train-split) mean forecast
# error counts as an anomaly. Reduced from 3.0 to 2.0 for more sensitive
# detection based on performance analysis.
FORECAST_THRESHOLD_K = 2.0


# -----------------------
# Utility metric functions
# -----------------------
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())

    prec = TP / (TP + FP + 1e-9)
    rec = TP / (TP + FN + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "TP": TP, "FP": FP, "FN": FN, "TN": TN}


def mean_time_to_detect(timestamps: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """
    For each ground-truth anomaly event (consecutive indices where y_true==1),
    find the first index j >= event_start where y_pred[j] == 1. Record (t_j - t_start).
    Return mean over detected events (misses are excluded from the average,
    since an undetected event has no defined detection latency).
    """
    ts = np.asarray(timestamps)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    N = len(ts)
    i = 0
    latencies = []

    while i < N:
        if y_true[i] == 1:
            start_t = ts[i]
            j = i
            while j < N and y_true[j] == 1:
                j += 1
            k = i
            while k < N:
                if y_pred[k] == 1:
                    latencies.append(float(ts[k] - start_t))
                    break
                k += 1
            i = j
        else:
            i += 1

    if len(latencies) == 0:
        return None
    return float(np.mean(latencies))


# -----------------------
# Core detection pipeline
# -----------------------
def ensure_models_ready():
    """Train models if artifacts missing. Both now train on split=='train'
    rows only (see lstm_forecast.py / autoencoder.py), so if you change the
    preprocessing pipeline, delete the old model files before re-running."""
    if not (os.path.exists(LSTM_MODEL_PATH) and os.path.exists(LSTM_SCALER_PATH)):
        print("[INFO] LSTM artifacts not found; training LSTM (this may take a while)...")
        train_lstm()
    if not (os.path.exists(AE_MODEL_PATH) and os.path.exists(AE_META_PATH)):
        print("[INFO] Autoencoder artifacts not found; training autoencoder...")
        train_autoencoder()


def load_processed_csv(path: str = LSTM_PROCESSED_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["is_anomaly", "is_vibration_anomaly", "is_temperature_anomaly"]:
        if col not in df.columns:
            df[col] = 0
    if "split" not in df.columns:
        raise ValueError(
            "Processed CSV has no 'split' column. Re-run data_preprocessor.py "
            "(the updated version) before running detection."
        )
    return df


def run_detection_and_metrics(
    processed_csv_path: str = LSTM_PROCESSED_CSV,
    out_csv: str = OUT_RESULTS_CSV,
    forecast_threshold_k: float = FORECAST_THRESHOLD_K,
):
    """Run detection using LSTM & Autoencoder, compute metrics on the TEST
    split only, and save full results (all rows) for inspection."""

    ensure_models_ready()

    df = load_processed_csv(processed_csv_path)
    timestamps = df["timestamp"].values
    features = df[[c for c in LSTM_FEATURES]].values.astype(np.float32)
    split = df["split"].values  # aligned 1:1 with df rows

    # ---- LSTM forecast errors (windowed) ----
    X_windowed = []
    for i in range(LSTM_WINDOW_SIZE, len(features)):
        X_windowed.append(features[i - LSTM_WINDOW_SIZE : i])
    X_windowed = np.asarray(X_windowed, dtype=np.float32)
    if len(X_windowed) == 0:
        raise ValueError("Not enough data for LSTM windowing.")

    # windowed sample j predicts row (j + LSTM_WINDOW_SIZE); track which
    # split that target row belongs to so we can fit the threshold on
    # train-only errors while still scoring every row for the output CSV.
    windowed_split = split[LSTM_WINDOW_SIZE:]

    lstm_model = keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    scaler = np.load(LSTM_SCALER_PATH, allow_pickle=True)
    X_mean, X_std = scaler["X_mean"], scaler["X_std"]
    y_mean, y_std = scaler["y_mean"], scaler["y_std"]

    X_s = (X_windowed - X_mean) / X_std
    preds_s = lstm_model.predict(X_s, verbose=0)
    preds = preds_s * y_std + y_mean

    actuals = features[LSTM_WINDOW_SIZE:]
    forecast_errors = np.linalg.norm(actuals - preds, axis=1)

    # Fit the forecast-error threshold ONLY on rows belonging to the train
    # split. This is the key leakage fix: previously mean/std came from the
    # full (anomaly-containing) error array, which inflates the threshold
    # and makes the detector less sensitive to real anomalies in test data.
    train_mask_windowed = (windowed_split == "train")
    if train_mask_windowed.sum() == 0:
        raise ValueError("No train-split rows available after windowing; check WINDOW_SIZE vs train_fraction.")
    fe_train_errors = forecast_errors[train_mask_windowed]
    fe_mean = float(fe_train_errors.mean())
    fe_std = float(fe_train_errors.std() if fe_train_errors.std() > 0 else 1.0)
    forecast_threshold = fe_mean + forecast_threshold_k * fe_std

    forecast_pred_windowed = (forecast_errors > forecast_threshold).astype(int)

    # ---- Autoencoder reconstruction errors with feature-specific detection ----
    # detect_anomalies() loads the threshold that was fit on train-split
    # data inside train_autoencoder(), so the mask it returns is already
    # correctly calibrated on clean data — no further threshold-fitting
    # needed here.
    ae_results = detect_anomalies(features[LSTM_WINDOW_SIZE:].tolist())
    ae_errors = ae_results["errors"]
    ae_threshold = float(ae_results["threshold"])
    ae_pred_windowed = ae_results["mask"].astype(int)
    
    # Extract feature-specific information
    feature_errors = ae_results["feature_errors"]  # shape: (n_samples, 3)
    feature_thresholds = ae_results["feature_thresholds"]
    feature_masks = ae_results["feature_masks"]  # shape: (n_samples, 3)
    feature_names = ae_results["feature_names"]
    
    # Feature-specific detection: temperature is index 0, vibration is index 1
    temp_idx = feature_names.index("temperature") if "temperature" in feature_names else 0
    vib_idx = feature_names.index("vibration") if "vibration" in feature_names else 1
    
    ae_temp_pred_windowed = feature_masks[:, temp_idx].astype(int)
    ae_vib_pred_windowed = feature_masks[:, vib_idx].astype(int)

    # ---- Confidence-based hybrid detection ----
    # Instead of simple binary OR/AND, use normalized confidence scores
    # for more nuanced detection. Higher confidence = more certain anomaly.
    
    # Forecast confidence: how many σ above threshold
    forecast_confidence = np.maximum(0, (forecast_errors - forecast_threshold) / (fe_std + 1e-9))
    
    # Autoencoder confidence: how many σ above threshold
    ae_mean = ae_threshold / 3.0  # approximate, since threshold = mean + 2*std
    ae_std = (ae_threshold - ae_mean) / 2.0
    ae_confidence = np.maximum(0, (ae_errors - ae_threshold) / (ae_std + 1e-9))
    
    # Combined confidence: max of the two (more sensitive than AND, more specific than blind OR)
    combined_confidence = np.maximum(forecast_confidence, ae_confidence)
    
    # Feature-specific confidences for temperature and vibration
    if feature_thresholds is not None:
        temp_threshold = feature_thresholds[temp_idx]
        vib_threshold = feature_thresholds[vib_idx]
        
        # Estimate std from threshold (threshold ≈ mean + 2*std)
        temp_std = temp_threshold / 3.0
        vib_std = vib_threshold / 3.0
        
        temp_confidence = np.maximum(0, (feature_errors[:, temp_idx] - temp_threshold) / (temp_std + 1e-9))
        vib_confidence = np.maximum(0, (feature_errors[:, vib_idx] - vib_threshold) / (vib_std + 1e-9))
    else:
        temp_confidence = ae_confidence
        vib_confidence = forecast_confidence
    
    # Primary hybrid detector: flag if confidence > 0 (exceeds threshold)
    combined_confidence_pred_windowed = (combined_confidence > 0).astype(int)
    
    # Alternative: weighted combination favoring forecast for vibration, AE for temperature
    weighted_vib_confidence = np.maximum(forecast_confidence * 1.2, vib_confidence)
    weighted_temp_confidence = np.maximum(ae_confidence * 1.2, temp_confidence)
    weighted_combined_confidence = np.maximum(weighted_vib_confidence, weighted_temp_confidence)
    weighted_combined_pred_windowed = (weighted_combined_confidence > 0).astype(int)
    
    # Keep legacy OR/AND for comparison
    combined_or_windowed = np.maximum(forecast_pred_windowed, ae_pred_windowed)
    combined_and_windowed = np.minimum(forecast_pred_windowed, ae_pred_windowed)

    # Pad back to full df length (rows before LSTM_WINDOW_SIZE have no
    # windowed prediction; they're necessarily part of train split anyway)
    def pad(arr_windowed):
        full = np.zeros(len(df), dtype=int)
        full[LSTM_WINDOW_SIZE:] = arr_windowed
        return full

    forecast_pred_full = pad(forecast_pred_windowed)
    ae_pred_full = pad(ae_pred_windowed)
    ae_temp_pred_full = pad(ae_temp_pred_windowed)
    ae_vib_pred_full = pad(ae_vib_pred_windowed)
    combined_or_full = pad(combined_or_windowed)
    combined_and_full = pad(combined_and_windowed)
    combined_confidence_pred_full = pad(combined_confidence_pred_windowed)
    weighted_combined_pred_full = pad(weighted_combined_pred_windowed)
    
    # Pad confidence scores (using 0.0 for non-windowed rows)
    def pad_float(arr_windowed):
        full = np.zeros(len(df), dtype=np.float32)
        full[LSTM_WINDOW_SIZE:] = arr_windowed
        return full
    
    forecast_confidence_full = pad_float(forecast_confidence)
    ae_confidence_full = pad_float(ae_confidence)
    combined_confidence_full = pad_float(combined_confidence)
    temp_confidence_full = pad_float(temp_confidence)
    vib_confidence_full = pad_float(vib_confidence)

    # ---- Ground truth ----
    y_true_combined = df["is_anomaly"].values.astype(int)
    y_true_vib = df["is_vibration_anomaly"].values.astype(int)
    y_true_temp = df["is_temperature_anomaly"].values.astype(int)

    # ---- Evaluate ONLY on the test split ----
    # This is the second leakage fix: metrics are no longer computed over
    # the whole dataset (which mixes clean train rows with the only rows
    # that actually contain anomalies). Evaluating on test-only avoids
    # diluting/inflating accuracy with trivially-correct train rows.
    test_mask = (split == "test")

    def test_slice(arr):
        return arr[test_mask]

    # Primary hybrid detector: confidence-based
    combined_metrics = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_confidence_pred_full))
    
    # Alternative detectors for comparison
    combined_metrics_weighted = compute_binary_metrics(test_slice(y_true_combined), test_slice(weighted_combined_pred_full))
    combined_metrics_or = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_or_full))
    combined_metrics_and = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_and_full))
    
    # Component detectors
    forecast_only_metrics = compute_binary_metrics(test_slice(y_true_vib), test_slice(forecast_pred_full))
    ae_only_metrics = compute_binary_metrics(test_slice(y_true_temp), test_slice(ae_pred_full))
    
    # Feature-specific autoencoder detectors
    ae_temp_only_metrics = compute_binary_metrics(test_slice(y_true_temp), test_slice(ae_temp_pred_full))
    ae_vib_only_metrics = compute_binary_metrics(test_slice(y_true_vib), test_slice(ae_vib_pred_full))
    
    # Hybrid detector performance on specific anomaly types
    vib_combined_metrics = compute_binary_metrics(test_slice(y_true_vib), test_slice(combined_confidence_pred_full))
    temp_combined_metrics = compute_binary_metrics(test_slice(y_true_temp), test_slice(combined_confidence_pred_full))

    # MTTD only makes sense within a contiguous, time-ordered slice, so we
    # compute it over the test split's own timestamps/labels.
    test_ts = timestamps[test_mask]
    mttd_vib = mean_time_to_detect(test_ts, test_slice(y_true_vib), test_slice(combined_confidence_pred_full))
    mttd_temp = mean_time_to_detect(test_ts, test_slice(y_true_temp), test_slice(combined_confidence_pred_full))

    # ---- Save full detailed results (all rows, for inspection) ----
    out_df = df.copy()
    out_df["forecast_error"] = 0.0
    out_df["ae_error"] = 0.0
    out_df.loc[LSTM_WINDOW_SIZE:, "forecast_error"] = forecast_errors
    out_df.loc[LSTM_WINDOW_SIZE:, "ae_error"] = ae_errors
    
    # Predictions from various detectors
    out_df["y_pred_forecast"] = forecast_pred_full
    out_df["y_pred_ae"] = ae_pred_full
    out_df["y_pred_ae_temp"] = ae_temp_pred_full
    out_df["y_pred_ae_vib"] = ae_vib_pred_full
    out_df["y_pred_combined_confidence"] = combined_confidence_pred_full
    out_df["y_pred_combined_weighted"] = weighted_combined_pred_full
    out_df["y_pred_combined_or"] = combined_or_full
    out_df["y_pred_combined_and"] = combined_and_full
    
    # Confidence scores
    out_df["forecast_confidence"] = forecast_confidence_full
    out_df["ae_confidence"] = ae_confidence_full
    out_df["combined_confidence"] = combined_confidence_full
    out_df["temp_confidence"] = temp_confidence_full
    out_df["vib_confidence"] = vib_confidence_full
    
    # Thresholds for reference
    out_df["forecast_threshold"] = forecast_threshold
    out_df["ae_threshold"] = ae_threshold
    if feature_thresholds is not None:
        out_df["temp_threshold"] = feature_thresholds[temp_idx]
        out_df["vib_threshold"] = feature_thresholds[vib_idx]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    def print_metrics(title, m):
        print(f"\n=== {title} ===")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall:    {m['recall']:.4f}")
        print(f"F1-score:  {m['f1']:.4f}")
        print(f"Accuracy:  {m['accuracy']:.4f}")
        print(f"TP/FP/FN/TN: {m['TP']}/{m['FP']}/{m['FN']}/{m['TN']}")

    print(f"\n[Evaluated on {test_mask.sum()} test-split rows only]")
    print(f"Forecast threshold (fit on train-split errors): {forecast_threshold:.4f}")
    print(f"Autoencoder threshold (fit on train-split errors): {ae_threshold:.4f}")
    if feature_thresholds is not None:
        print(f"Temperature feature threshold: {feature_thresholds[temp_idx]:.4f}")
        print(f"Vibration feature threshold: {feature_thresholds[vib_idx]:.4f}")

    print_metrics("Combined Confidence-Based (PRIMARY) — hybrid detector with confidence scoring", combined_metrics)
    print_metrics("Combined Weighted — alternative weighted confidence approach", combined_metrics_weighted)
    print_metrics("Combined OR (legacy) — simple binary OR", combined_metrics_or)
    print_metrics("Combined AND (legacy) — strict binary AND", combined_metrics_and)
    
    print("\n--- Component Detector Performance ---")
    print_metrics("Forecast-only vs vibration anomalies", forecast_only_metrics)
    print_metrics("Autoencoder-only vs temperature anomalies", ae_only_metrics)
    print_metrics("AE feature-specific (temperature) vs temperature anomalies", ae_temp_only_metrics)
    print_metrics("AE feature-specific (vibration) vs vibration anomalies", ae_vib_only_metrics)
    
    print("\n--- Primary Detector Performance by Anomaly Type ---")
    print_metrics("Confidence-based hybrid vs vibration anomalies", vib_combined_metrics)
    print_metrics("Confidence-based hybrid vs temperature anomalies", temp_combined_metrics)

    print(f"\nMean Time to Detect (Vibration, confidence-based detector): {mttd_vib}")
    print(f"Mean Time to Detect (Temperature, confidence-based detector): {mttd_temp}")
    print(f"\nDetailed detection results saved to: {out_csv}")

    return {
        "combined_metrics": combined_metrics,
        "combined_metrics_weighted": combined_metrics_weighted,
        "combined_metrics_or": combined_metrics_or,
        "combined_metrics_and": combined_metrics_and,
        "forecast_only_metrics": forecast_only_metrics,
        "ae_only_metrics": ae_only_metrics,
        "ae_temp_only_metrics": ae_temp_only_metrics,
        "ae_vib_only_metrics": ae_vib_only_metrics,
        "vib_combined_metrics": vib_combined_metrics,
        "temp_combined_metrics": temp_combined_metrics,
        "mttd_vib": mttd_vib,
        "mttd_temp": mttd_temp,
        "out_csv": out_csv,
    }


if __name__ == "__main__":
    run_detection_and_metrics()