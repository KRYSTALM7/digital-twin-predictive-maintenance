import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import itertools

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

# Default sigma multiplier (will be calibrated on validation set)
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


def create_validation_split(df: pd.DataFrame, validation_fraction: float = 0.15) -> pd.DataFrame:
    """
    Carve out validation set from training data for threshold calibration.
    
    Maintains temporal ordering: takes last 15% of training data as validation.
    This avoids data leakage while preserving time-series structure.
    
    Split structure after this function:
    - train: 85% of original training data (clean, for model training)
    - val: 15% of original training data (clean, for threshold calibration)
    - test: unchanged (anomalies, for final evaluation)
    
    Args:
        df: DataFrame with 'split' column containing 'train' and 'test'
        validation_fraction: Fraction of training data to use for validation
    
    Returns:
        DataFrame with 'split' column containing 'train', 'val', and 'test'
    """
    df_copy = df.copy()
    
    train_mask = df_copy['split'] == 'train'
    train_indices = df_copy[train_mask].index.tolist()
    
    # Take last validation_fraction of training data as validation
    n_train = len(train_indices)
    n_val = int(n_train * validation_fraction)
    
    if n_val == 0:
        raise ValueError(f"Not enough training data to create validation split. "
                         f"Training samples: {n_train}, validation fraction: {validation_fraction}")
    
    # Split indices: first part stays train, last part becomes validation
    val_start_idx = n_train - n_val
    val_indices = train_indices[val_start_idx:]
    
    # Update split column
    df_copy.loc[val_indices, 'split'] = 'val'
    
    train_count = (df_copy['split'] == 'train').sum()
    val_count = (df_copy['split'] == 'val').sum()
    test_count = (df_copy['split'] == 'test').sum()
    
    print(f"[INFO] Split created: Train={train_count}, Val={val_count}, Test={test_count}")
    
    return df_copy


def _smart_hybrid_detection_with_thresholds(
    fc: np.ndarray,  # forecast_confidence
    ac: np.ndarray,  # ae_confidence  
    tc: np.ndarray,  # temp_confidence
    vc: np.ndarray,  # vib_confidence
    strong_conf: float,
    moderate_conf: float,
    weak_conf: float
) -> np.ndarray:
    """
    SMART hybrid detection logic with configurable thresholds.
    
    Detection strategies:
    1. Strong signal from any single source (OR logic for obvious faults)
    2. Moderate coupled fault (both temp and vibration elevated)
    3. Multiple weak signals corroborate (at least 2 detectors)
    4. Forecast + AE both see something (original confidence rule)
    """
    n = len(fc)
    flags = np.zeros(n, dtype=int)
    
    for i in range(n):
        # Strategy 1: Strong single signal
        if (tc[i] > strong_conf or vc[i] > strong_conf or 
            fc[i] > strong_conf or ac[i] > strong_conf):
            flags[i] = 1
        
        # Strategy 2: Moderate coupled fault
        elif tc[i] > moderate_conf and vc[i] > moderate_conf:
            flags[i] = 1
        
        # Strategy 3: Multiple weak signals
        elif sum([
            tc[i] > weak_conf,
            vc[i] > weak_conf,
            fc[i] > weak_conf,
            ac[i] > weak_conf
        ]) >= 2:
            flags[i] = 1
        
        # Strategy 4: Forecast + AE agreement
        elif fc[i] > weak_conf and ac[i] > weak_conf:
            flags[i] = 1
    
    return flags


def calibrate_smart_thresholds(
    forecast_confidence_val: np.ndarray,
    ae_confidence_val: np.ndarray,
    temp_confidence_val: np.ndarray,
    vib_confidence_val: np.ndarray,
    y_val: np.ndarray,
    threshold_ranges: Dict[str, Tuple[float, float, float]] = None
) -> Dict[str, float]:
    """
    Grid search over confidence thresholds on validation set to maximize F1.
    
    This is the scientifically rigorous way to set thresholds: systematically
    search over reasonable ranges and pick the combination that maximizes
    validation F1. Never touches test set.
    
    Args:
        *_confidence_val: Confidence scores on validation set
        y_val: Ground truth labels on validation set
        threshold_ranges: Dict mapping threshold names to (min, max, step) tuples
    
    Returns:
        Dict of optimized thresholds: {'strong': X, 'moderate': Y, 'weak': Z}
    """
    if threshold_ranges is None:
        # Default search ranges based on previous analysis
        threshold_ranges = {
            'strong': (0.3, 1.0, 0.1),      # Search 0.3, 0.4, ..., 1.0
            'moderate': (0.15, 0.5, 0.05),   # Search 0.15, 0.20, ..., 0.50
            'weak': (0.05, 0.3, 0.05)        # Search 0.05, 0.10, ..., 0.30
        }
    
    print("[INFO] Calibrating SMART thresholds on validation set...")
    print(f"[INFO] Search ranges: {threshold_ranges}")
    
    # Generate all combinations
    strong_vals = np.arange(*threshold_ranges['strong'])
    moderate_vals = np.arange(*threshold_ranges['moderate'])
    weak_vals = np.arange(*threshold_ranges['weak'])
    
    best_f1 = -1.0
    best_thresholds = None
    best_metrics = None
    
    total_combinations = len(strong_vals) * len(moderate_vals) * len(weak_vals)
    print(f"[INFO] Testing {total_combinations} threshold combinations...")
    
    for strong, moderate, weak in itertools.product(strong_vals, moderate_vals, weak_vals):
        # Ensure threshold ordering: strong >= moderate >= weak
        if not (strong >= moderate >= weak):
            continue
        
        # Apply SMART detection logic with these thresholds
        predictions = _smart_hybrid_detection_with_thresholds(
            forecast_confidence_val,
            ae_confidence_val,
            temp_confidence_val,
            vib_confidence_val,
            strong, moderate, weak
        )
        
        # Compute F1
        metrics = compute_binary_metrics(y_val, predictions)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_thresholds = {'strong': float(strong), 'moderate': float(moderate), 'weak': float(weak)}
            best_metrics = metrics
    
    print(f"[INFO] Best thresholds: {best_thresholds}")
    print(f"[INFO] Validation F1: {best_f1:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    
    return best_thresholds


def run_detection_and_metrics(
    processed_csv_path: str = LSTM_PROCESSED_CSV,
    out_csv: str = OUT_RESULTS_CSV,
    forecast_threshold_k: float = FORECAST_THRESHOLD_K,
    calibrate_thresholds: bool = False,
    run_ablation: bool = False,
    validation_fraction: float = 0.15,
    threshold_ranges: Dict[str, Tuple[float, float, float]] = None,
):
    """
    Run detection using LSTM & Autoencoder with optional threshold calibration and ablation study.
    
    Pipeline modes:
    1. Standard mode (calibrate_thresholds=False, run_ablation=False):
       - Uses train/test split from preprocessing
       - Uses default thresholds (strong=0.5, moderate=0.25, weak=0.15)
       - Evaluates only on test set
    
    2. Calibration mode (calibrate_thresholds=True):
       - Creates train/val/test split
       - Performs grid search on validation set
       - Uses calibrated thresholds on test set
    
    3. Ablation mode (run_ablation=True):
       - Requires calibration mode
       - Compares SMART against 7 baseline methods
       - All methods use same data splits
    
    Args:
        processed_csv_path: Path to preprocessed data
        out_csv: Path to save detailed results
        forecast_threshold_k: Default forecast error threshold multiplier (if not calibrating)
        calibrate_thresholds: Whether to calibrate thresholds on validation set
        run_ablation: Whether to run comprehensive ablation study
        validation_fraction: Fraction of training data for validation (default: 0.15)
        threshold_ranges: Custom threshold search ranges for calibration
    """
    
    if run_ablation and not calibrate_thresholds:
        raise ValueError("Ablation study requires threshold calibration (calibrate_thresholds=True)")
    
    ensure_models_ready()

    df = load_processed_csv(processed_csv_path)
    
    # Create validation split if calibration or ablation requested
    if calibrate_thresholds or run_ablation:
        df = create_validation_split(df, validation_fraction=validation_fraction)
    
    timestamps = df["timestamp"].values
    features = df[[c for c in LSTM_FEATURES]].values.astype(np.float32)
    split = df["split"].values  # 'train', 'val' (if created), or 'test'

    # ---- LSTM forecast errors (windowed) ----
    X_windowed = []
    for i in range(LSTM_WINDOW_SIZE, len(features)):
        X_windowed.append(features[i - LSTM_WINDOW_SIZE : i])
    X_windowed = np.asarray(X_windowed, dtype=np.float32)
    if len(X_windowed) == 0:
        raise ValueError("Not enough data for LSTM windowing.")

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

    # Fit forecast-error threshold on train split only
    train_mask_windowed = (windowed_split == "train")
    if train_mask_windowed.sum() == 0:
        raise ValueError("No train-split rows available after windowing; check WINDOW_SIZE vs train_fraction.")
    fe_train_errors = forecast_errors[train_mask_windowed]
    fe_mean = float(fe_train_errors.mean())
    fe_std = float(fe_train_errors.std() if fe_train_errors.std() > 0 else 1.0)
    forecast_threshold = fe_mean + forecast_threshold_k * fe_std

    forecast_pred_windowed = (forecast_errors > forecast_threshold).astype(int)

    # ---- Autoencoder reconstruction errors with feature-specific detection ----
    ae_results = detect_anomalies(features[LSTM_WINDOW_SIZE:].tolist())
    ae_errors = ae_results["errors"]
    ae_threshold = float(ae_results["threshold"])
    ae_pred_windowed = ae_results["mask"].astype(int)
    
    # Extract feature-specific information
    feature_errors = ae_results["feature_errors"]
    feature_thresholds = ae_results["feature_thresholds"]
    feature_masks = ae_results["feature_masks"]
    feature_names = ae_results["feature_names"]
    
    temp_idx = feature_names.index("temperature") if "temperature" in feature_names else 0
    vib_idx = feature_names.index("vibration") if "vibration" in feature_names else 1
    
    ae_temp_pred_windowed = feature_masks[:, temp_idx].astype(int)
    ae_vib_pred_windowed = feature_masks[:, vib_idx].astype(int)

    # ---- Calculate confidence scores for all detectors ----
    forecast_confidence = np.maximum(0, (forecast_errors - forecast_threshold) / (fe_std + 1e-9))
    
    ae_mean = ae_threshold / 3.0
    ae_std = (ae_threshold - ae_mean) / 2.0
    ae_confidence = np.maximum(0, (ae_errors - ae_threshold) / (ae_std + 1e-9))
    
    combined_confidence = np.maximum(forecast_confidence, ae_confidence)
    
    if feature_thresholds is not None:
        temp_threshold = feature_thresholds[temp_idx]
        vib_threshold = feature_thresholds[vib_idx]
        
        temp_std = temp_threshold / 3.0
        vib_std = vib_threshold / 3.0
        
        temp_confidence = np.maximum(0, (feature_errors[:, temp_idx] - temp_threshold) / (temp_std + 1e-9))
        vib_confidence = np.maximum(0, (feature_errors[:, vib_idx] - vib_threshold) / (vib_std + 1e-9))
    else:
        temp_confidence = ae_confidence
        vib_confidence = forecast_confidence
    
    # ---- Threshold Calibration (if requested) ----
    if calibrate_thresholds:
        val_mask_windowed = (windowed_split == "val")
        if val_mask_windowed.sum() == 0:
            raise ValueError("No validation split rows available after windowing.")
        
        y_val = df["is_anomaly"].values[LSTM_WINDOW_SIZE:][val_mask_windowed].astype(int)
        
        calibrated_thresholds = calibrate_smart_thresholds(
            forecast_confidence[val_mask_windowed],
            ae_confidence[val_mask_windowed],
            temp_confidence[val_mask_windowed],
            vib_confidence[val_mask_windowed],
            y_val,
            threshold_ranges=threshold_ranges
        )
        
        # Use calibrated thresholds
        STRONG_CONF = calibrated_thresholds['strong']
        MODERATE_CONF = calibrated_thresholds['moderate']
        WEAK_CONF = calibrated_thresholds['weak']
    else:
        # Use default thresholds
        STRONG_CONF = 0.5
        MODERATE_CONF = 0.25
        WEAK_CONF = 0.15
    
    print(f"[INFO] Using thresholds: STRONG={STRONG_CONF:.3f}, MODERATE={MODERATE_CONF:.3f}, WEAK={WEAK_CONF:.3f}")
    
    # ---- Apply SMART hybrid detection with chosen thresholds ----
    combined_smart_pred_windowed = _smart_hybrid_detection_with_thresholds(
        forecast_confidence,
        ae_confidence,
        temp_confidence,
        vib_confidence,
        STRONG_CONF,
        MODERATE_CONF,
        WEAK_CONF
    )
    
    # Alternative detectors for comparison
    combined_confidence_pred_windowed = (combined_confidence > 0).astype(int)
    weighted_vib_confidence = np.maximum(forecast_confidence * 1.2, vib_confidence)
    weighted_temp_confidence = np.maximum(ae_confidence * 1.2, temp_confidence)
    weighted_combined_confidence = np.maximum(weighted_vib_confidence, weighted_temp_confidence)
    weighted_combined_pred_windowed = (weighted_combined_confidence > 0).astype(int)
    
    combined_or_windowed = np.maximum(forecast_pred_windowed, ae_pred_windowed)
    combined_and_windowed = np.minimum(forecast_pred_windowed, ae_pred_windowed)

    # Pad back to full df length
    def pad(arr_windowed):
        full = np.zeros(len(df), dtype=int)
        full[LSTM_WINDOW_SIZE:] = arr_windowed
        return full

    forecast_pred_full = pad(forecast_pred_windowed)
    ae_pred_full = pad(ae_pred_windowed)
    ae_temp_pred_full = pad(ae_temp_pred_windowed)
    ae_vib_pred_full = pad(ae_vib_pred_windowed)
    combined_smart_pred_full = pad(combined_smart_pred_windowed)
    combined_or_full = pad(combined_or_windowed)
    combined_and_full = pad(combined_and_windowed)
    combined_confidence_pred_full = pad(combined_confidence_pred_windowed)
    weighted_combined_pred_full = pad(weighted_combined_pred_windowed)
    
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
    test_mask = (split == "test")

    def test_slice(arr):
        return arr[test_mask]

    # Primary hybrid detector: SMART
    combined_metrics = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_smart_pred_full))
    
    # Alternative detectors for comparison
    combined_metrics_old_confidence = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_confidence_pred_full))
    combined_metrics_weighted = compute_binary_metrics(test_slice(y_true_combined), test_slice(weighted_combined_pred_full))
    combined_metrics_or = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_or_full))
    combined_metrics_and = compute_binary_metrics(test_slice(y_true_combined), test_slice(combined_and_full))
    
    # Component detectors
    forecast_only_metrics = compute_binary_metrics(test_slice(y_true_vib), test_slice(forecast_pred_full))
    ae_only_metrics = compute_binary_metrics(test_slice(y_true_temp), test_slice(ae_pred_full))
    ae_temp_only_metrics = compute_binary_metrics(test_slice(y_true_temp), test_slice(ae_temp_pred_full))
    ae_vib_only_metrics = compute_binary_metrics(test_slice(y_true_vib), test_slice(ae_vib_pred_full))
    
    # Detector performance by anomaly type
    vib_combined_metrics = compute_binary_metrics(test_slice(y_true_vib), test_slice(combined_smart_pred_full))
    temp_combined_metrics = compute_binary_metrics(test_slice(y_true_temp), test_slice(combined_smart_pred_full))

    # MTTD
    test_ts = timestamps[test_mask]
    mttd_vib = mean_time_to_detect(test_ts, test_slice(y_true_vib), test_slice(combined_smart_pred_full))
    mttd_temp = mean_time_to_detect(test_ts, test_slice(y_true_temp), test_slice(combined_smart_pred_full))

    # ---- Run Ablation Study (if requested) ----
    ablation_results = None
    if run_ablation:
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE ABLATION STUDY")
        print("="*70)
        
        try:
            from experiments.ablation_study import run_comprehensive_ablation_study
            
            ablation_results = run_comprehensive_ablation_study(
                features=features[LSTM_WINDOW_SIZE:],
                labels=y_true_combined[LSTM_WINDOW_SIZE:],
                split=windowed_split,
                forecast_errors=forecast_errors,
                ae_errors=ae_errors,
                feature_confidences={
                    'forecast': forecast_confidence,
                    'ae': ae_confidence,
                    'temp': temp_confidence,
                    'vib': vib_confidence
                },
                smart_hybrid_predictions=combined_smart_pred_windowed
            )
            
            print("\n" + "="*70)
            print("ABLATION STUDY COMPLETE")
            print("="*70)
            print(ablation_results.to_string(index=False))
            
        except ImportError as e:
            print(f"\n[WARNING] Could not import ablation_study module: {e}")
            print("[WARNING] Skipping ablation study.")
    
    # ---- Save full detailed results (all rows, for inspection) ----
    out_df = df.copy()
    out_df["forecast_error"] = 0.0
    out_df["ae_error"] = 0.0
    out_df.loc[LSTM_WINDOW_SIZE:, "forecast_error"] = forecast_errors
    out_df.loc[LSTM_WINDOW_SIZE:, "ae_error"] = ae_errors
    
    out_df["y_pred_forecast"] = forecast_pred_full
    out_df["y_pred_ae"] = ae_pred_full
    out_df["y_pred_ae_temp"] = ae_temp_pred_full
    out_df["y_pred_ae_vib"] = ae_vib_pred_full
    out_df["y_pred_combined_smart"] = combined_smart_pred_full
    out_df["y_pred_combined_confidence"] = combined_confidence_pred_full
    out_df["y_pred_combined_weighted"] = weighted_combined_pred_full
    out_df["y_pred_combined_or"] = combined_or_full
    out_df["y_pred_combined_and"] = combined_and_full
    
    out_df["forecast_confidence"] = forecast_confidence_full
    out_df["ae_confidence"] = ae_confidence_full
    out_df["combined_confidence"] = combined_confidence_full
    out_df["temp_confidence"] = temp_confidence_full
    out_df["vib_confidence"] = vib_confidence_full
    
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

    print_metrics("Combined SMART Hybrid (PRIMARY)", combined_metrics)
    print_metrics("Combined Confidence-Based (OLD)", combined_metrics_old_confidence)
    print_metrics("Combined Weighted", combined_metrics_weighted)
    print_metrics("Combined OR (legacy)", combined_metrics_or)
    print_metrics("Combined AND (legacy)", combined_metrics_and)
    
    print("\n--- Component Detector Performance ---")
    print_metrics("Forecast-only vs vibration anomalies", forecast_only_metrics)
    print_metrics("Autoencoder-only vs temperature anomalies", ae_only_metrics)
    print_metrics("AE feature-specific (temperature)", ae_temp_only_metrics)
    print_metrics("AE feature-specific (vibration)", ae_vib_only_metrics)
    
    print("\n--- SMART Detector by Anomaly Type ---")
    print_metrics("SMART vs vibration anomalies", vib_combined_metrics)
    print_metrics("SMART vs temperature anomalies", temp_combined_metrics)

    print(f"\nMean Time to Detect (Vibration, SMART): {mttd_vib}")
    print(f"\Mean Time to Detect (Temperature, SMART): {mttd_temp}")
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
        "ablation_results": ablation_results,
        "calibrated_thresholds": {
            'strong': STRONG_CONF,
            'moderate': MODERATE_CONF,
            'weak': WEAK_CONF
        } if calibrate_thresholds else None,
    }


if __name__ == "__main__":
    run_detection_and_metrics()
