import os
import csv
import random
from typing import List, Dict, Any

RAW_CSV_PATH = os.path.join("data", "raw", "simulated_spindle_data.csv")
PROCESSED_CSV_PATH = os.path.join("data", "processed", "spindle_data_with_anomalies.csv")

# Fraction of the (time-ordered) data used for training. Training data is
# NEVER anomaly-injected — it represents "known normal" operation, which is
# what the LSTM/autoencoder thresholds should be calibrated against.
TRAIN_FRACTION = 0.7


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _coerce_row_types(row: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(row)
    for k in ["timestamp", "temperature", "vibration", "current"]:
        try:
            coerced[k] = float(coerced[k]) if coerced[k] not in (None, "", "NaN") else None
        except Exception:
            coerced[k] = None
    # ground_truth_failure comes from the simulation now; carry it through
    # as an int, defaulting to 0 if the raw file predates this column
    # (keeps backward compatibility with older simulation CSVs).
    try:
        coerced["ground_truth_failure"] = int(float(coerced.get("ground_truth_failure", 0)))
    except Exception:
        coerced["ground_truth_failure"] = 0
    return coerced


def _clean_missing_values(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric_fields = ["timestamp", "temperature", "vibration", "current"]
    typed_rows = [_coerce_row_types(r) for r in rows]

    seeds: Dict[str, float] = {}
    for field in numeric_fields:
        for r in typed_rows:
            if r.get(field) is not None:
                seeds[field] = float(r[field])
                break
        if field not in seeds:
            seeds[field] = 0.0

    last_values = dict(seeds)
    cleaned: List[Dict[str, Any]] = []
    for r in typed_rows:
        new_r = dict(r)
        for field in numeric_fields:
            if new_r.get(field) is None:
                new_r[field] = last_values[field]
            else:
                last_values[field] = float(new_r[field])
        cleaned.append(new_r)
    return cleaned


def _normalize_timestamps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return rows
    t0 = float(rows[0]["timestamp"]) if rows[0].get("timestamp") is not None else 0.0
    normalized: List[Dict[str, Any]] = []
    for r in rows:
        new_r = dict(r)
        ts = float(new_r.get("timestamp", 0.0))
        new_r["timestamp"] = max(ts - t0, 0.0)
        normalized.append(new_r)
    return normalized


def _initialize_anomaly_flags(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for row in data:
        row["is_anomaly"] = 0
        row["is_vibration_anomaly"] = 0
        row["is_temperature_anomaly"] = 0
    return data


def _split_train_test(data: List[Dict[str, Any]], train_fraction: float = TRAIN_FRACTION):
    """Time-ordered split: first `train_fraction` of rows are TRAIN (clean),
    remainder are TEST (eligible for anomaly injection). Splitting by time
    (not randomly) avoids leaking future information into training, and
    keeps the split reproducible regardless of anomaly injection method."""
    n = len(data)
    split_idx = int(n * train_fraction)
    for i, row in enumerate(data):
        row["split"] = "train" if i < split_idx else "test"
    return data, split_idx


def inject_anomalies(
    data: List[Dict[str, Any]],
    method: str = "physics",  # Changed from "genai" to "physics"
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Inject physically motivated anomalies ONLY into rows marked split == 'test'.
    
    Physics-based fault types:
    1. Bearing wear: gradual coupled temp+vib increase (realistic degradation)
    2. Cooling failure: temperature drift without vibration (thermal system fault)
    3. Sudden imbalance: vibration spike + slight temp (mechanical fault)
    
    This replaces random/LLM-based injection with physically meaningful faults
    that reflect real spindle failure modes.
    
    Args:
        data: List of data rows with 'split' column
        method: Anomaly injection method (currently only 'physics' supported)
        seed: Random seed for reproducibility (default: 42)
    """
    random.seed(seed)

    augmented = [dict(r) for r in data]
    test_indices = [i for i, r in enumerate(augmented) if r.get("split") == "test"]
    n_test = len(test_indices)
    if n_test == 0:
        return augmented

    def clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(value, max_v))

    print(f"[INFO] Injecting physics-based anomalies into {n_test} test rows.")

    # -----------------------------
    # Fault Type 1: Bearing Wear (Coupled, Gradual)
    # Physically realistic: bearing degradation causes both friction (heat) and vibration
    # -----------------------------
    num_bearing_faults = max(1, n_test // 400)
    print(f"[INFO] Injecting {num_bearing_faults} bearing wear fault(s)")
    
    for _ in range(num_bearing_faults):
        start = random.randint(0, max(0, n_test - 40))
        length = random.randint(25, 45)  # Longer duration for gradual fault
        
        for offset in range(length):
            if start + offset >= n_test:
                break
            real_i = test_indices[start + offset]
            progress = offset / (length - 1)  # 0 to 1
            
            # Coupled increase: both temperature and vibration rise together
            temp = float(augmented[real_i]["temperature"])
            vib = float(augmented[real_i]["vibration"])
            
            # Gradual temperature increase (4-8°C over duration)
            temp_increase = progress * random.uniform(4.0, 8.0)
            # Gradual vibration increase (1.8-3.0x over duration)
            vib_factor = 1.0 + progress * random.uniform(0.8, 2.0)
            
            augmented[real_i]["temperature"] = clamp(temp + temp_increase, -50.0, 300.0)
            augmented[real_i]["vibration"] = clamp(vib * vib_factor, 0.0, 5.0)
            augmented[real_i]["is_anomaly"] = 1
            augmented[real_i]["is_temperature_anomaly"] = 1
            augmented[real_i]["is_vibration_anomaly"] = 1

    # -----------------------------
    # Fault Type 2: Cooling System Failure (Temperature Only)
    # Physically realistic: thermal system degrades, mechanical system unaffected
    # -----------------------------
    num_cooling_faults = max(1, n_test // 450)
    print(f"[INFO] Injecting {num_cooling_faults} cooling failure fault(s)")
    
    for _ in range(num_cooling_faults):
        start = random.randint(0, max(0, n_test - 35))
        length = random.randint(20, 40)
        drift_total = random.uniform(5.0, 10.0)  # 5-10°C drift
        
        for offset in range(length):
            if start + offset >= n_test:
                break
            real_i = test_indices[start + offset]
            progress = offset / (length - 1)
            
            temp = float(augmented[real_i]["temperature"])
            augmented[real_i]["temperature"] = clamp(temp + drift_total * progress, -50.0, 300.0)
            augmented[real_i]["is_anomaly"] = 1
            augmented[real_i]["is_temperature_anomaly"] = 1
            # Vibration remains normal (cooling failure doesn't affect mechanical balance)

    # -----------------------------
    # Fault Type 3: Sudden Imbalance (Vibration Spike + Slight Temp)
    # Physically realistic: mechanical imbalance causes immediate vibration, slight friction increase
    # -----------------------------
    num_imbalance_faults = max(1, n_test // 350)
    print(f"[INFO] Injecting {num_imbalance_faults} imbalance fault(s)")
    
    for _ in range(num_imbalance_faults):
        start = random.randint(0, max(0, n_test - 10))
        length = random.randint(3, 10)
        
        for offset in range(length):
            if start + offset >= n_test:
                break
            real_i = test_indices[start + offset]
            
            vib = float(augmented[real_i]["vibration"])
            temp = float(augmented[real_i]["temperature"])
            
            # Sharp vibration increase (3.5-5.5x)
            augmented[real_i]["vibration"] = clamp(vib * random.uniform(3.5, 5.5), 0.0, 5.0)
            # Slight temperature increase from additional friction (1-3°C)
            augmented[real_i]["temperature"] = clamp(temp + random.uniform(1.0, 3.0), -50.0, 300.0)
            augmented[real_i]["is_anomaly"] = 1
            augmented[real_i]["is_vibration_anomaly"] = 1
            # Temperature increase is minor, don't mark as temperature anomaly unless you want to

    print(f"[INFO] Physics-based anomaly injection complete.")
    return augmented


def _save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    header = [
        "timestamp",
        "state",
        "temperature",
        "vibration",
        "current",
        "ground_truth_failure",
        "split",
        "is_anomaly",
        "is_vibration_anomaly",
        "is_temperature_anomaly",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        if not rows:
            return
        for r in rows:
            writer.writerow({k: r.get(k) for k in header})


def preprocess_data(
    raw_csv_path: str = RAW_CSV_PATH,
    processed_csv_path: str = PROCESSED_CSV_PATH,
    anomaly_method: str = "physics",  # Changed default to "physics"
    train_fraction: float = TRAIN_FRACTION,
    seed: int = 42,
) -> str:
    """Run full preprocessing pipeline and save processed CSV.

    Adds a 'split' column ('train' / 'test'). Only the test portion ever
    receives injected anomalies, so downstream model thresholds trained on
    split == 'train' are calibrated on clean data only.
    
    Now uses physics-based anomaly injection by default for more realistic
    fault patterns.
    
    Args:
        raw_csv_path: Path to raw simulation data
        processed_csv_path: Path to save processed data
        anomaly_method: Anomaly injection method ('physics')
        train_fraction: Fraction of data for training (default: 0.7)
        seed: Random seed for reproducible anomaly generation (default: 42)
    """
    data = _load_csv(raw_csv_path)
    data = _clean_missing_values(data)
    data = _normalize_timestamps(data)
    data = _initialize_anomaly_flags(data)
    data, split_idx = _split_train_test(data, train_fraction=train_fraction)

    print(f"[INFO] Train rows: {split_idx} | Test rows: {len(data) - split_idx}")

    data = inject_anomalies(data, method=anomaly_method, seed=seed)

    out_dir = os.path.dirname(processed_csv_path)
    if out_dir:
        _ensure_dir(out_dir)
    _save_csv(processed_csv_path, data)
    return processed_csv_path


if __name__ == "__main__":
    path = preprocess_data()
    print(f"Preprocessing complete. CSV saved to: {path}")