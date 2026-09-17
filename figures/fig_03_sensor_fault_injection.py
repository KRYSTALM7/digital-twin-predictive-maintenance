"""
Generate Figure 3: Sensor Data with Physics-Motivated Fault Injection
Publication-quality time-series visualization showing normal operation and injected faults.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory if it doesn't exist
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Read detection results
df = pd.read_csv("outputs/detection_results.csv")

# Filter to test split only (where faults are injected)
df_test = df[df['split'] == 'test'].copy()

# Configure publication-quality plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Identify fault periods
fault_mask = df_test['ground_truth_failure'] == 1
fault_periods = []
in_fault = False
start_idx = None

for idx, is_fault in enumerate(fault_mask):
    if is_fault and not in_fault:
        start_idx = idx
        in_fault = True
    elif not is_fault and in_fault:
        fault_periods.append((df_test.iloc[start_idx]['timestamp'], 
                             df_test.iloc[idx-1]['timestamp']))
        in_fault = False

# Handle case where fault extends to end
if in_fault:
    fault_periods.append((df_test.iloc[start_idx]['timestamp'], 
                         df_test.iloc[-1]['timestamp']))

# Plot temperature
ax1.plot(df_test['timestamp'], df_test['temperature'], 
         linewidth=1.0, color='#D62728', label='Temperature', zorder=2)
ax1.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold')
ax1.grid(True, alpha=0.15, linestyle='--')
ax1.set_ylim([65, 90])

# Highlight fault periods on temperature plot (reduced opacity)
for start, end in fault_periods:
    ax1.axvspan(start, end, alpha=0.10, color='red', label='_nolegend_', zorder=1)

# Plot vibration
ax2.plot(df_test['timestamp'], df_test['vibration'], 
         linewidth=1.0, color='#1F77B4', label='Vibration', zorder=2)
ax2.set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Vibration (mm/s)', fontsize=10, fontweight='bold')
ax2.grid(True, alpha=0.15, linestyle='--')
ax2.set_ylim([0, 0.14])

# Highlight fault periods on vibration plot (reduced opacity)
for start, end in fault_periods:
    ax2.axvspan(start, end, alpha=0.10, color='red', label='_nolegend_', zorder=1)

# Add legend with fault period label (only once, on top panel)
if fault_periods:
    ax1.axvspan(fault_periods[0][0], fault_periods[0][1], 
                alpha=0.10, color='red', label='Fault Period', zorder=1)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)

ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)

plt.tight_layout()

# Save PNG only
output_png = output_dir / "fig_03_sensor_fault_injection.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"✓ Figure 3 saved: {output_png}")
print(f"  Format: PNG (300 DPI)")
print(f"  Fault periods detected: {len(fault_periods)}")

plt.close()
