# Publication-Quality Figure Generation

This directory contains Python scripts to generate all publication-quality figures for the research paper.

## Prerequisites

Ensure you have the required packages installed:
```bash
pip install pandas matplotlib numpy scipy
```

## Scripts

1. **fig_03_sensor_fault_injection.py** - Sensor data time-series with fault periods
2. **fig_04_smart_detection_example.py** - Multi-panel SMART detection example
3. **fig_05_ablation_f1_comparison.py** - Horizontal bar chart comparing 8 methods
4. **fig_06_multiseed_f1_distribution.py** - Box plot showing 10-seed distributions

## Generate All Figures

Run from the project root directory (`digital_twin/`):

```bash
python figures/fig_03_sensor_fault_injection.py
python figures/fig_04_smart_detection_example.py
python figures/fig_05_ablation_f1_comparison.py
python figures/fig_06_multiseed_f1_distribution.py
```

Or run all at once (PowerShell):
```powershell
Get-ChildItem figures\*.py | ForEach-Object { python $_.FullName }
```

## Expected Outputs

Each script generates two files:
- **PNG** format (300 DPI) - for Word/Google Docs insertion
- **PDF** format (vector) - for LaTeX or high-quality printing

```
figures/
├── fig_03_sensor_fault_injection.png
├── fig_03_sensor_fault_injection.pdf
├── fig_04_smart_detection_example.png
├── fig_04_smart_detection_example.pdf
├── fig_05_ablation_f1_comparison.png
├── fig_05_ablation_f1_comparison.pdf
├── fig_06_multiseed_f1_distribution.png
└── fig_06_multiseed_f1_distribution.pdf
```

## Data Sources

- **Figure 3:** `outputs/detection_results.csv` (test split)
- **Figure 4:** `outputs/detection_results.csv` (timesteps 735-765)
- **Figure 5:** `outputs/ablation_study.csv` (8 methods)
- **Figure 6:** `outputs/multi_run_validation.csv` (10 seeds)

## Paper Placement

### Section 5: Results and Discussion

- **Figure 3** → Section 5.1 (Simulation Data Quality)
- **Figure 5** → Section 5.2 (Ablation Study Results)
- **Figure 6** → Section 5.3 (Multi-Seed Validation)
- **Figure 4** → Section 5.4 (SMART Detection Mechanism)

## Notes

- All figures use consistent typography (serif font, 10pt base size)
- Color scheme is publication-friendly and colorblind-safe
- Both PNG (raster, 300 DPI) and PDF (vector) formats generated
- Scripts are independently runnable and use actual experimental data
- No manual data entry required - all values loaded from CSV files

## Troubleshooting

**If you get "ModuleNotFoundError: No module named 'scipy'":**
```bash
pip install scipy
```

**If paths don't work:**
Make sure you run the scripts from the project root directory (`digital_twin/`), not from inside the `figures/` directory.

**If figures look different in Word:**
Use the PNG files for Word documents. PDFs are for LaTeX or high-quality printing.
