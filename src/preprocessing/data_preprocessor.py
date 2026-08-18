import os
import csv
import random
from google import genai
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

load_dotenv()

RAW_CSV_PATH = os.path.join("data", "raw", "simulated_spindle_data.csv")
PROCESSED_CSV_PATH = os.path.join("data", "processed", "spindle_data_with_anomalies.csv")

# Fraction of the (time-ordered) data used for training. Training data is
# NEVER anomaly-injected — it represents "known normal" operation, which is
# what the LSTM/autoencoder thresholds should be calibrated against.
TRAIN_FRACTION = 0.7

# Sanity bounds for any anomaly severity, whether it comes from the random
# fallback or from a Gemini-proposed plan. This stops an LLM (or a bad
# random draw) from producing an absurd, non-reproducible anomaly magnitude.
MAX_TEMP_DRIFT_TOTAL = 8.0   # degrees C, matches random fallback's upper bound
MIN_TEMP_DRIFT_TOTAL = 3.0
VIBRATION_FACTOR_RANGE = (3.0, 5.0)  # unchanged from original


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_csv(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _get_genai_anomaly_plan_gemini(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Uses Gemini to decide anomaly injection windows (indices relative to
    the slice of data passed in — i.e. the TEST portion only, see below)."""
    api_key = os.getenv("GEMINI_API_KEY")
    print("Loaded key:", api_key is not None)

    if not api_key:
        print("[WARN] No GEMINI_API_KEY found. Using random anomalies.")
        return {}

    client = genai.Client(api_key=api_key)

    preview = data[:200]

    prompt = (
        "You are a predictive maintenance expert. Based on the CNC spindle sensor data, "
        "propose anomaly injection windows.\n"
        "Respond ONLY in pure JSON. No backticks. No text. No explanation.\n\n"
        "Return format:\n"
        "{\n"
        "  \"vibration_spikes\": [[start, length], ...],\n"
        "  \"temperature_drifts\": [[start, length, total_change], ...]\n"
        "}\n\n"
        f"DATA:\n{preview}"
    )

    try:
        # gemini-flash-latest is an auto-updating alias maintained by Google,
        # so this keeps working as models are retired/replaced upstream
        # (as of July 2026 it points to Gemini 3.5/3.6 Flash).
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        text = response.text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in response")
        return json.loads(text[start:end])
    except Exception as e:
        print("[ERROR] Gemini request failed. Falling back to random logic.", e)
        return {}


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
    method: str = "genai",
    seed: int | None = 42,
) -> List[Dict[str, Any]]:
    """Inject anomalies ONLY into rows already marked split == 'test'.
    `data` here is expected to be the full dataset (with 'split' column set);
    train rows are passed through untouched."""
    if seed is not None:
        random.seed(seed)

    augmented = [dict(r) for r in data]
    test_indices = [i for i, r in enumerate(augmented) if r.get("split") == "test"]
    n_test = len(test_indices)
    if n_test == 0:
        return augmented

    def clamp(value: float, min_v: float, max_v: float) -> float:
        return max(min_v, min(value, max_v))

    def clamp_drift_total(total: float) -> float:
        # Preserve sign (drift can be negative) but bound magnitude so a
        # bad LLM response can't produce an unrealistic swing.
        sign = 1.0 if total >= 0 else -1.0
        mag = clamp(abs(total), MIN_TEMP_DRIFT_TOTAL, MAX_TEMP_DRIFT_TOTAL)
        return sign * mag

    # test_indices maps a "local" position within the test slice to the
    # actual row index in `augmented`. Gemini only ever sees the test slice.
    test_rows_only = [augmented[i] for i in test_indices]

    if method == "genai":
        plan = _get_genai_anomaly_plan_gemini(test_rows_only)

        if plan:
            print("[INFO] Injecting anomalies using Gemini plan (test split only).")

            for start, length in plan.get("vibration_spikes", []):
                start = int(start)
                length = int(length)
                for local_i in range(start, min(n_test, start + length)):
                    real_i = test_indices[local_i]
                    vib = float(augmented[real_i]["vibration"])
                    factor = random.uniform(*VIBRATION_FACTOR_RANGE)
                    augmented[real_i]["vibration"] = clamp(vib * factor, 0.0, 5.0)
                    augmented[real_i]["is_anomaly"] = 1
                    augmented[real_i]["is_vibration_anomaly"] = 1

            for start, length, drift_total in plan.get("temperature_drifts", []):
                start = int(start)
                length = int(length)
                drift_total = clamp_drift_total(float(drift_total))
                step = drift_total / max(1, length - 1)
                delta = 0.0
                for local_i in range(start, min(n_test, start + length)):
                    real_i = test_indices[local_i]
                    temp = float(augmented[real_i]["temperature"])
                    augmented[real_i]["temperature"] = clamp(temp + delta, -50.0, 300.0)
                    delta += step
                    augmented[real_i]["is_anomaly"] = 1
                    augmented[real_i]["is_temperature_anomaly"] = 1

            return augmented

        print("[WARN] Gemini returned no plan. Defaulting to random anomalies.")

    # -----------------------------
    # Random fallback (test split only)
    # -----------------------------
    num_spike_windows = max(1, n_test // 250)
    num_drift_windows = max(1, n_test // 300)

    for _ in range(num_spike_windows):
        start = random.randint(0, max(0, n_test - 3))
        length = random.randint(2, 6)
        for local_i in range(start, min(n_test, start + length)):
            real_i = test_indices[local_i]
            vib = float(augmented[real_i]["vibration"])
            factor = random.uniform(*VIBRATION_FACTOR_RANGE)
            augmented[real_i]["vibration"] = clamp(vib * factor, 0.0, 5.0)
            augmented[real_i]["is_anomaly"] = 1
            augmented[real_i]["is_vibration_anomaly"] = 1

    for _ in range(num_drift_windows):
        start = random.randint(0, max(0, n_test - 15))
        length = random.randint(10, 30)
        drift_total = clamp_drift_total(random.uniform(MIN_TEMP_DRIFT_TOTAL, MAX_TEMP_DRIFT_TOTAL))
        step = drift_total / max(1, length - 1)
        delta = 0.0
        for local_i in range(start, min(n_test, start + length)):
            real_i = test_indices[local_i]
            temp = float(augmented[real_i]["temperature"])
            augmented[real_i]["temperature"] = clamp(temp + delta, -50.0, 300.0)
            delta += step
            augmented[real_i]["is_anomaly"] = 1
            augmented[real_i]["is_temperature_anomaly"] = 1

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
    anomaly_method: str = "genai",
    train_fraction: float = TRAIN_FRACTION,
) -> str:
    """Run full preprocessing pipeline and save processed CSV.

    Adds a 'split' column ('train' / 'test'). Only the test portion ever
    receives injected anomalies, so downstream model thresholds trained on
    split == 'train' are calibrated on clean data only.
    """
    data = _load_csv(raw_csv_path)
    data = _clean_missing_values(data)
    data = _normalize_timestamps(data)
    data = _initialize_anomaly_flags(data)
    data, split_idx = _split_train_test(data, train_fraction=train_fraction)

    print(f"[INFO] Train rows: {split_idx} | Test rows: {len(data) - split_idx}")

    data = inject_anomalies(data, method=anomaly_method)

    out_dir = os.path.dirname(processed_csv_path)
    if out_dir:
        _ensure_dir(out_dir)
    _save_csv(processed_csv_path, data)
    return processed_csv_path


if __name__ == "__main__":
    path = preprocess_data()
    print(f"Preprocessing complete. CSV saved to: {path}")