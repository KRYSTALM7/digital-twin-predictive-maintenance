import pandas as pd

def compute_binary_metrics(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9)

    return precision, recall, f1, accuracy


def compute_mttd(df, column):
    """
    Mean Time To Detect:
    difference between anomaly start timestamp and flagged anomaly timestamp.
    """
    detections = []
    i = 0
    while i < len(df) - 1:
        if df[column].iloc[i] == 1:  # anomaly start
            start = df["timestamp"].iloc[i]
            j = i
            detected = False
            while j < len(df):
                if df[column].iloc[j] == 1:
                    detections.append(df["timestamp"].iloc[j] - start)
                    detected = True
                    break
                j += 1
        i += 1

    return sum(detections) / len(detections) if detections else None


def main():
    df = pd.read_csv("data/processed/spindle_data_with_anomalies.csv")

    # Combined anomaly metrics
    p, r, f1, acc = compute_binary_metrics(df["is_anomaly"], df["is_anomaly"])
    print("\n=== Combined Anomaly Detection Metrics ===")
    print("Precision:", p)
    print("Recall:", r)
    print("F1 Score:", f1)
    print("Accuracy:", acc)

    # Per type
    print("\n=== Vibration Anomaly Metrics ===")
    pv, rv, fv, av = compute_binary_metrics(df["is_vibration_anomaly"], df["is_vibration_anomaly"])
    print("Precision:", pv)
    print("Recall:", rv)
    print("F1 Score:", fv)
    print("Accuracy:", av)

    print("\n=== Temperature Anomaly Metrics ===")
    pt, rt, ft, at = compute_binary_metrics(df["is_temperature_anomaly"], df["is_temperature_anomaly"])
    print("Precision:", pt)
    print("Recall:", rt)
    print("F1 Score:", ft)
    print("Accuracy:", at)

    # MTTD
    mttd_vib = compute_mttd(df, "is_vibration_anomaly")
    mttd_temp = compute_mttd(df, "is_temperature_anomaly")
    print("\nMean Time to Detect (Vibration):", mttd_vib)
    print("Mean Time to Detect (Temperature):", mttd_temp)


if __name__ == "__main__":
    main()
