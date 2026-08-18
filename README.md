# CNC Spindle Digital Twin - Predictive Maintenance System

**Authors**: Sujan Kumar MV, Ganesh Khekare

A comprehensive digital twin implementation for CNC (Computer Numerical Control) spindle predictive maintenance using hybrid deep learning models. This system combines discrete-event simulation, LSTM-based time series forecasting, and autoencoder-based anomaly detection to predict equipment failures before they occur.

## Overview

This project simulates a CNC spindle's operational lifecycle and uses machine learning to detect anomalies and predict failures in real-time. The system is designed for industrial predictive maintenance applications where minimizing unplanned downtime is critical.

### Key Features

- **Discrete-Event Simulation**: SimPy-based CNC spindle simulation with realistic failure modes, wear progression, and maintenance cycles
- **Advanced Hybrid Anomaly Detection**: Combines LSTM forecasting and autoencoder reconstruction with confidence-based scoring for superior detection
- **Feature-Specific Detection**: Per-feature thresholds for temperature and vibration enable targeted anomaly identification
- **Confidence-Based Scoring**: Weighted confidence approach replaces binary OR/AND logic for more nuanced detection (F1: 0.537)
- **Enhanced Autoencoder Architecture**: Improved 32→16→4→16→32 architecture with better capacity and sensitivity
- **LLM-Enhanced Data Augmentation**: Uses Google Gemini API to intelligently inject realistic anomalies into test data
- **Comprehensive Metrics**: Precision, recall, F1-score, and Mean Time to Detect (MTTD) for model evaluation
- **Feedback Loop Architecture**: Real-time decision-making system that integrates multiple detection signals

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CNC Spindle Simulation                       │
│              (SimPy: RUNNING → FAILURE → MAINTENANCE)           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Raw sensor data
                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                Data Preprocessing Pipeline                      │
│  • Clean/normalize timestamps                                   │
│  • Train/test split (70/30, time-ordered)                       │
│  • Gemini-based anomaly injection (test split only)             │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Processed data with ground truth
                      ↓
        ┌─────────────┴─────────────┐
        │                           │
        ↓                           ↓
┌──────────────────┐      ┌──────────────────┐
│  LSTM Forecaster │      │   Autoencoder    │
│  (Vibration)     │      │  (Temperature)   │
│                  │      │                  │
│  • 64 LSTM units │      │  • 32→16→4→16→32 │
│  • Window=20     │      │  • MSE loss      │
│  • 2σ threshold  │      │  • 2σ threshold  │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         │ Forecast errors         │ Reconstruction errors
         │ + confidence scores     │ + per-feature errors
         │                         │
         └──────────┬──────────────┘
                    ↓
         ┌────────────────────────┐
         │  Confidence-Based      │
         │  Hybrid Detector       │
         │  • Weighted scoring    │
         │  • Feature-specific    │
         │  • F1: 0.537           │
         └──────────┬─────────────┘
                    │
                    ↓
         ┌─────────────────────┐
         │  Decision Engine    │
         │  • Normal           │
         │  • Monitor          │
         │  • Maintenance Req. │
         └─────────────────────┘
```

## Project Structure

```
digital_twin/
├── data/
│   ├── raw/                          # Raw simulation output
│   └── processed/                    # Preprocessed with anomalies
├── src/
│   ├── simulation/                   # CNC spindle discrete-event simulation
│   ├── preprocessing/                # Data cleaning & anomaly injection
│   ├── models/                       # LSTM & Autoencoder models
│   └── feedback/                     # Detection pipeline & metrics
├── scripts/                          # Test & visualization scripts
├── outputs/                          # Trained models & results
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Google Gemini API key (optional, for LLM-enhanced anomaly injection)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/KRYSTALM7/digital-twin-predictive-maintenance.git
cd digital_twin
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

*Note: The system falls back to random anomaly injection if no API key is provided.*

## Quick Start

Run the complete pipeline:

```bash
# 1. Generate simulated sensor data
python -m src.simulation.simpy_simulation

# 2. Preprocess and inject anomalies
python -m src.preprocessing.data_preprocessor

# 3. Train models
python -m src.models.lstm_forecast
python -m src.models.autoencoder

# 4. Run detection and evaluation
python -m src.feedback.detection_and_metrics

# 5. Generate real-time feedback decisions
python -m src.feedback.feedback_loop
```

## Technical Highlights

### Models
- **LSTM Forecaster**: Time-series prediction with 64 LSTM units for vibration anomaly detection (2σ sensitivity threshold)
- **Autoencoder**: Enhanced architecture (32→16→4→16→32) for temperature anomaly detection with 2σ threshold
- **Confidence-Based Hybrid Detector**: Weighted confidence scoring replaces binary OR/AND logic
  - Primary method: Confidence-based (F1: 0.495)
  - Weighted variant: Feature-specific weighting (F1: 0.537)
  - Per-feature thresholds enable targeted detection

### Performance Improvements (vs. Previous Version)
- **Overall F1-score**: 0.263 → 0.537 (104% improvement)
- **Recall**: 0.185 → 0.537 (190% improvement)
- **Temperature Detection**: 0.000 → 0.435 F1 (previously non-functional)
- **Detection Latency**: 9.5 → 8.5 time units (11% faster)
- **Precision**: Maintained at 0.537 while dramatically improving recall

### Data Pipeline
- Time-ordered 70/30 train/test split to prevent data leakage
- Thresholds calibrated on clean training data only
- Gemini API integration for realistic anomaly injection

### Metrics
- Precision, Recall, F1-score, Accuracy
- Mean Time to Detect (MTTD)
- Evaluated on test split only for fair assessment

## Data Description

### Sensor Data Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Normalized simulation time (seconds from start) |
| `state` | string | Spindle state: RUNNING, IDLE, FAILURE, MAINTENANCE |
| `temperature` | float | Spindle temperature (°C), baseline ~70 |
| `vibration` | float | Vibration amplitude (mm/s), baseline ~0.02 |
| `current` | float | Motor current (A), baseline ~10 + wear |
| `ground_truth_failure` | int | 1 if state==FAILURE, else 0 |
| `split` | string | 'train' (70%, clean) or 'test' (30%, with anomalies) |
| `is_anomaly` | int | 1 if any anomaly type injected |
| `is_vibration_anomaly` | int | 1 if vibration spike injected |
| `is_temperature_anomaly` | int | 1 if temperature drift injected |

### Anomaly Types

1. **Vibration Spikes**: Sudden 3-5× increase in vibration amplitude over 2-6 timesteps
2. **Temperature Drifts**: Gradual temperature increase of 3-8°C over 10-30 timesteps

## Model Details

### LSTM Forecaster

- **Architecture**: 64 LSTM units → 32 Dense → 3 output features
- **Input**: 20-timestep sliding window
- **Output**: Next timestep prediction [temperature, vibration, current]
- **Training**: MSE loss, Adam optimizer (lr=1e-3), 20 epochs
- **Anomaly Detection**: Forecast error > (train_mean + 2×train_std)
- **Improvements**: More sensitive threshold (3σ→2σ)

### Autoencoder

- **Architecture**: 3 → 32 → 16 → 4 (bottleneck) → 16 → 32 → 3
- **Training**: MSE loss, 100 epochs, batch size 32
- **Anomaly Detection**: 
  - Overall reconstruction error > (train_mean + 2×train_std)
  - Per-feature errors for temperature and vibration
  - Feature-specific thresholds enable targeted detection
- **Improvements**: Increased capacity (bottleneck 2→4), more training (50→100 epochs), more sensitive threshold (3σ→2σ)

### Hybrid Detector

**Primary Method (Confidence-Based Scoring)**:
- Computes normalized confidence scores (σ above threshold) for each detector
- Combined confidence = max(forecast_confidence, ae_confidence)
- Flags anomaly if combined_confidence > 0
- More nuanced than binary OR/AND logic

**Weighted Variant (Best Performance)**:
- Applies domain-specific weights: forecast favored for vibration (1.2×), AE for temperature (1.2×)
- Achieves F1: 0.537 vs 0.495 for standard confidence-based
- Used for final reported metrics

**Legacy Methods (Comparison)**:
- **OR logic**: Anomaly if *either* detector flags (high recall, lower precision)
- **AND logic**: Anomaly if *both* detectors flag (high precision, low recall)

**Decision Rules**:
- **Maintenance Required**: High confidence anomaly (> 1.0σ)
- **Monitor**: Moderate confidence anomaly (0-1.0σ)
- **Normal Operation**: No anomaly detected

## Evaluation Metrics

The system reports the following metrics on the **test split only**:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total
- **Mean Time to Detect (MTTD)**: Average time from anomaly start to first detection

Results are saved to `outputs/detection_results.csv` with per-row predictions, confidence scores, and errors.

### Latest Results (Improved System)

**Weighted Confidence-Based Detector (Primary)**:
- Precision: 0.537
- Recall: 0.537
- F1-Score: 0.537
- Accuracy: 0.834
- MTTD (Vibration): 0.0 (immediate detection)
- MTTD (Temperature): 8.5 time units

**Component Performance**:
- Autoencoder (Temperature): F1 0.435 (previously 0.000)
- AE Feature-Specific (Temperature): F1 0.364
- AE Feature-Specific (Vibration): F1 0.487

**Comparison with Legacy Methods**:
- Previous OR method: F1 0.263
- Confidence-based: F1 0.495 (88% improvement)
- Weighted confidence: F1 0.537 (104% improvement)

## Key Design Decisions

### Train/Test Split Strategy
- **Time-ordered** 70/30 split (not random) to prevent data leakage
- **Train split**: Always clean, no anomalies injected
- **Test split**: Contains injected anomalies for evaluation
- Prevents the model from "memorizing" anomalies it should detect

### Threshold Calibration
- All detection thresholds (LSTM forecast error, autoencoder reconstruction error) are fit using **train split statistics only**
- This simulates real-world deployment where thresholds are calibrated on known-normal operation
- Previous versions leaked test anomalies into threshold computation, inflating thresholds and reducing sensitivity

### LLM-Enhanced Anomaly Injection
- Gemini API analyzes sensor patterns to suggest realistic anomaly windows
- Provides domain-aware injection that better mimics real failure modes
- Sanity bounds (3-8°C drift, 3-5× vibration factor) prevent unrealistic LLM suggestions
- Falls back to random injection if API unavailable

### Reproducibility
- Random seed set to `42` in preprocessing for deterministic anomaly placement
- All model training uses fixed random initialization
- SimPy simulation uses exponential distribution for failure times (pseudo-random, reproducible with seed)

## Output Files

- `outputs/detection_results.csv`: Full per-row predictions, errors, and ground truth
- `outputs/feedback_log.csv`: Real-time decision log (Maintenance/Monitor/Normal)
- `outputs/lstm_model.h5`: Trained LSTM model weights
- `outputs/autoencoder_model.h5`: Trained autoencoder weights
- `outputs/*_scaler.npz`: Normalization parameters for inference

## Dependencies

Core dependencies (see `requirements.txt`):
- `tensorflow` - Deep learning models
- `simpy` - Discrete-event simulation
- `numpy` - Numerical computing
- `google-genai` - Gemini API integration
- `python-dotenv` - Environment variable management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure:
1. All tests pass before submitting PRs
2. Code follows existing style conventions
3. Documentation is updated for API changes

## Citation

If you use this work in academic research, please cite:

```bibtex
@software{cnc_digital_twin,
  title={CNC Spindle Digital Twin - Predictive Maintenance System},
  author={Ghimire, Sujan and Khekare, Ganesh},
  year={2026},
  url={https://github.com/KRYSTALM7/digital-twin-predictive-maintenance}
}
```

---

**Note**: This is a research/educational project. For production deployment in industrial settings, additional validation, safety analysis, and domain expert review are required.
