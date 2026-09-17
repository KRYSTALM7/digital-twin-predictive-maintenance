"""
Generate Figure 4: SMART Detection Example
Multi-panel visualization showing sensor behavior, anomaly scores, and detection output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Read detection results
df = pd.read_csv("outputs/detection_results.csv")

# Select window around fault at timesteps 743-756 (±10 timesteps for context)
window_start = 735
window_end = 765
df_window = df[(df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)].copy()

# Configure publication-quality plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Create 3-panel figure
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# PANEL A: Sensor Data
ax1_twin = ax1.twinx()

# Plot temperature and vibration
line1 = ax1.plot(df_window['timestamp'], df_window['temperature'], 
                 linewidth=1.2, color='#D62728', label='Temperature', alpha=0.9)
line2 = ax1_twin.plot(df_window['timestamp'], df_window['vibration'], 
                      linewidth=1.2, color='#1F77B4', label='Vibration', alpha=0.9)

# Shade fault period
fault_mask = df_window['ground_truth_failure'] == 1
if fault_mask.any():
    fault_start = df_window[fault_mask].iloc[0]['timestamp']
    fault_end = df_window[fault_mask].iloc[-1]['timestamp']
    ax1.axvspan(fault_start, fault_end, alpha=0.12, color='red', label='_nolegend_', zorder=0)

ax1.set_ylabel('Temperature (°C)', color='#D62728', fontsize=10, fontweight='bold')
ax1_twin.set_ylabel('Vibration (mm/s)', color='#1F77B4', fontsize=10, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#D62728')
ax1_twin.tick_params(axis='y', labelcolor='#1F77B4')
ax1.grid(True, alpha=0.15, linestyle='--')

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=9, framealpha=0.95)

# PANEL B: Anomaly Scores
ax2.plot(df_window['timestamp'], df_window['forecast_error'], 
         linewidth=1.0, color='#FF7F0E', label='LSTM Forecast Error', alpha=0.85)
ax2.plot(df_window['timestamp'], df_window['ae_error'] * 1000,  # Scale for visibility
         linewidth=1.0, color='#2CA02C', label='AE Reconstruction Error (×10³)', alpha=0.85)

# Plot thresholds
forecast_thresh = df_window['forecast_threshold'].iloc[0]
ae_thresh = df_window['ae_threshold'].iloc[0] * 1000

ax2.axhline(forecast_thresh, color='#FF7F0E', linestyle='--', linewidth=0.8, alpha=0.6)
ax2.axhline(ae_thresh, color='#2CA02C', linestyle='--', linewidth=0.8, alpha=0.6)

ax2.set_ylabel('Error Magnitude', fontsize=10, fontweight='bold')
ax2.set_ylim([-0.5, max(df_window['forecast_error'].max(), ae_thresh * 1.5)])
ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.15, linestyle='--')

# PANEL C: Detection Output
# Plot ground truth as filled area
ax3.fill_between(df_window['timestamp'], 0, df_window['ground_truth_failure'], 
                 step='mid', alpha=0.3, color='red', label='Ground Truth')

# Plot SMART predictions as step function
ax3.step(df_window['timestamp'], df_window['y_pred_combined_smart'], 
         where='mid', linewidth=2.0, color='darkblue', label='SMART Detection', alpha=0.9)

ax3.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Detection Flag', fontsize=10, fontweight='bold')
ax3.set_ylim([-0.1, 1.3])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Normal', 'Anomaly'])
ax3.legend(loc='upper left', fontsize=9, framealpha=0.95)
ax3.grid(True, alpha=0.15, linestyle='--', axis='x')

plt.tight_layout()

# Save PNG only
output_png = output_dir / "fig_04_smart_detection_example.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"✓ Figure 4 saved: {output_png}")
print(f"  Format: PNG (300 DPI)")
print(f"  Window: timesteps {window_start}-{window_end}")

plt.close()
