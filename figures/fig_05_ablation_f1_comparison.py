"""
Generate Figure 5: Ablation Study F1-Score Comparison
Horizontal bar chart comparing all 8 detection methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Read ablation study results
df = pd.read_csv("outputs/ablation_study.csv")

# Sort by F1-score (descending)
df = df.sort_values('f1', ascending=True)  # Ascending for horizontal bars (bottom to top)

# Configure publication-quality plot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Create figure
fig, ax = plt.subplots(figsize=(10, 5))

# Create color array (highlight SMART in distinct color)
colors = ['#1F77B4' if 'SMART' not in method else '#D62728' 
          for method in df['method']]

# Create horizontal bar chart
bars = ax.barh(df['method'], df['f1'], color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for idx, (f1_val, method) in enumerate(zip(df['f1'], df['method'])):
    ax.text(f1_val + 0.005, idx, f'{f1_val:.3f}', 
            va='center', fontsize=9, fontweight='bold')

# Formatting
ax.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_xlim([0, max(df['f1']) * 1.15])
ax.set_ylabel('Method', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.15, linestyle='--', axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save PNG only
output_png = output_dir / "fig_05_ablation_f1_comparison.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')

print(f"✓ Figure 5 saved: {output_png}")
print(f"  Format: PNG (300 DPI)")
print(f"\nF1-Score Rankings:")
for method, f1 in zip(df['method'][::-1], df['f1'][::-1]):
    print(f"  {method:30s}: {f1:.4f}")

plt.close()
