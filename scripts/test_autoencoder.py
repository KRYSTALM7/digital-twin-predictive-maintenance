import numpy as np
import pandas as pd
from src.models.autoencoder import detect_anomalies

df = pd.read_csv("data/processed/spindle_data_with_anomalies.csv")
sample = df[["temperature", "vibration", "current"]].iloc[:5].values.astype(np.float32)

# Inject one artificial anomaly
sample[2, 1] = 10.0  # exaggerate vibration

result = detect_anomalies(sample)
print("Mask:", result["mask"])
print("Errors:", result["errors"])
print("Threshold:", result["threshold"])
