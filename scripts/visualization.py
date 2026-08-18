import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/feedback_log.csv")

plt.figure(figsize=(12,6))
plt.plot(df["timestamp"], df["forecast_error"], label="Forecast Error")
plt.axhline(df["forecast_threshold"].iloc[0], color="r", linestyle="--", label="Forecast Threshold")

plt.scatter(df["timestamp"][df["decision"]=="Maintenance Required"], 
            df["forecast_error"][df["decision"]=="Maintenance Required"],
            color="red", label="Maintenance Required", zorder=3)

plt.scatter(df["timestamp"][df["decision"]=="Monitor"], 
            df["forecast_error"][df["decision"]=="Monitor"],
            color="orange", label="Monitor", zorder=3)

plt.title("Feedback Loop Decision Timeline")
plt.xlabel("Timestamp")
plt.ylabel("Forecast Error")
plt.legend()
plt.show()
