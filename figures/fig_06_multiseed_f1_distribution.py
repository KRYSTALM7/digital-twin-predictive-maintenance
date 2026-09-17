"""
Generate Figure 6: Multi-Seed F1-Score Distribution
Box plot with individual observations showing statistical robustness across 10 seeds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Read multi-run validation results
df = pd.read_csv("outputs/multi_run_validation.csv")

# Extract F1 scores for each method
smart_f1 = df['smart_f1'].values
lstm_f1 = df['lstm_f1'].values
ae_f1 = df['ae_f1'].values

# Calculate means
smart_mean = np.mean(smart_f1)
lstm_mean = np.mean(lstm_f1)
ae_mean = np.mean(ae_f1)

# Configure publication-quality plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Prepare data for box plot
data_to_plot = [smart_f1, lstm_f1, ae_f1]
positions = [1, 2, 3]
labels = ['SMART\nHybrid', 'LSTM\nOnly', 'Autoencoder\nOnly']
colors = ['#D62728', '#FF7F0E', '#2CA02C']

# Create box plots
bp = ax.boxplot(data_to_plot, positions=positions, widths=0.5,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.2),
                whiskerprops=dict(color='black', linewidth=1.2),
                capprops=dict(color='black', linewidth=1.2),
                medianprops=dict(color='black', linewidth=1.5))

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Overlay individual seed observations
np.random.seed(42)
for idx, (data, pos, color) in enumerate(zip(data_to_plot, positions, colors)):
    # Add jitter for visibility
    jittered_x = pos + np.random.normal(0, 0.04, size=len(data))
    ax.scatter(jittered_x, data, alpha=0.7, s=60, color=color, 
               edgecolors='black', linewidths=0.8, zorder=3)

# Add mean markers
for pos, mean_val, color in zip(positions, [smart_mean, lstm_mean, ae_mean], colors):
    ax.scatter(pos, mean_val, marker='D', s=100, color=color, 
               edgecolors='black', linewidths=1.2, zorder=4, label='_nolegend_')

# Formatting
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Detection Method', fontsize=12, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim([0, 0.7])
ax.grid(True, alpha=0.2, linestyle='--', axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Simplified legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
           markeredgecolor='black', markersize=10, alpha=0.6, label='Distribution (IQR)'),
    Line2D([0], [0], color='black', linewidth=1.5, label='Median'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
           markeredgecolor='black', markersize=8, alpha=0.7, label='Individual Seeds'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
           markeredgecolor='black', markersize=8, label='Mean')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

plt.tight_layout()

# Save PNG only
output_png = output_dir / "fig_06_multiseed_f1_distribution.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"✓ Figure 6 saved: {output_png}")
print(f"  Format: PNG (300 DPI)")
print(f"  SMART mean F1: {smart_mean:.4f}")
print(f"  LSTM mean F1:  {lstm_mean:.4f}")
print(f"  AE mean F1:    {ae_mean:.4f}")

plt.close()
