"""
Experimental configuration for publication-quality validation.

All hyperparameters and experimental settings are centralized here
to ensure reproducibility and prevent test-set tuning.
"""

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
SIMULATION_DURATION = 1000.0  # seconds
SIMULATION_TIMESTEP = 1.0  # seconds
MEAN_TIME_TO_FAILURE = 300.0  # seconds
MEAN_REPAIR_TIME = 50.0  # seconds
PREVENTIVE_MAINTENANCE_INTERVAL = 200.0  # seconds

# Base sensor parameters (for normal operation)
BASE_TEMPERATURE = 70.0  # °C
BASE_TEMPERATURE_STD = 2.0  # °C
BASE_VIBRATION = 0.02  # mm/s
BASE_VIBRATION_STD = 0.005  # mm/s
BASE_CURRENT = 10.0  # A
BASE_CURRENT_STD = 0.5  # A

# ============================================================================
# DATA SPLIT PARAMETERS
# ============================================================================
TRAIN_FRACTION = 0.70  # 70% for training
VAL_FRACTION = 0.15  # 15% for validation (from training set)
TEST_FRACTION = 0.30  # 30% for final evaluation

# Time-ordered split ensures no future information leaks into training
SPLIT_METHOD = "time_ordered"  # vs "random" (not recommended for time series)

# ============================================================================
# ANOMALY INJECTION PARAMETERS
# ============================================================================
# These ranges are based on typical spindle bearing failure signatures
# from literature (see paper citations)

# Bearing wear (coupled temperature + vibration increase)
BEARING_WEAR_DURATION_RANGE = (25, 45)  # timesteps
BEARING_WEAR_TEMP_INCREASE_RANGE = (4.0, 8.0)  # °C
BEARING_WEAR_VIB_FACTOR_RANGE = (1.8, 3.0)  # multiplier
BEARING_WEAR_FREQUENCY = 400  # inject 1 per N test timesteps

# Cooling system failure (temperature drift only)
COOLING_FAILURE_DURATION_RANGE = (20, 40)  # timesteps
COOLING_FAILURE_TEMP_DRIFT_RANGE = (5.0, 10.0)  # °C
COOLING_FAILURE_FREQUENCY = 450  # inject 1 per N test timesteps

# Mechanical imbalance (vibration spike + slight temperature increase)
IMBALANCE_DURATION_RANGE = (3, 10)  # timesteps
IMBALANCE_VIB_FACTOR_RANGE = (3.5, 5.5)  # multiplier
IMBALANCE_TEMP_INCREASE_RANGE = (1.0, 3.0)  # °C
IMBALANCE_FREQUENCY = 350  # inject 1 per N test timesteps

# Random seed for reproducible anomaly injection
ANOMALY_INJECTION_SEED = 42

# ============================================================================
# MODEL HYPERPARAMETERS (Fixed - No Test-Set Tuning)
# ============================================================================

# LSTM Forecaster
LSTM_UNITS = 64
LSTM_WINDOW_SIZE = 20  # timesteps lookback
LSTM_DENSE_UNITS = 32
LSTM_LEARNING_RATE = 1e-3
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 64

# Autoencoder
AE_ARCHITECTURE = [32, 16, 4, 16, 32]  # encoder → bottleneck → decoder
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 100
AE_BATCH_SIZE = 32

# Feature names (consistent across all models)
FEATURE_NAMES = ["temperature", "vibration", "current"]

# ============================================================================
# THRESHOLD CALIBRATION PARAMETERS
# ============================================================================
# Thresholds will be calibrated on VALIDATION split only
# Test set is never used for threshold selection

THRESHOLD_CALIBRATION_METHOD = "grid_search"  # vs "percentile" or "roc"

# Grid search ranges (for validation-based calibration)
STRONG_CONF_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MODERATE_CONF_RANGE = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
WEAK_CONF_RANGE = [0.05, 0.10, 0.15, 0.20, 0.25]

# Number of standard deviations for baseline threshold methods
SIGMA_MULTIPLIER_LSTM = 2.0  # for LSTM forecast error threshold
SIGMA_MULTIPLIER_AE = 2.0  # for AE reconstruction error threshold

# ============================================================================
# MULTI-RUN VALIDATION PARAMETERS
# ============================================================================
N_EXPERIMENTAL_RUNS = 10  # number of independent runs for variance estimation
RANDOM_SEEDS = list(range(N_EXPERIMENTAL_RUNS))  # [0, 1, 2, ..., 9]

# Statistical significance level
ALPHA = 0.05  # for p-value threshold in hypothesis tests

# ============================================================================
# EVALUATION METRICS
# ============================================================================
PRIMARY_METRIC = "f1"  # optimization objective
SECONDARY_METRICS = ["precision", "recall", "accuracy"]

# Mean Time To Detect (MTTD) computed separately for each anomaly type
COMPUTE_MTTD = True

# ============================================================================
# BASELINE METHODS (for fair comparison)
# ============================================================================
# External baselines from sklearn
EXTERNAL_BASELINES = {
    "isolation_forest": {
        "contamination": "auto",  # estimated from data
        "n_estimators": 100,
        "random_state": 42
    },
    "one_class_svm": {
        "nu": "auto",  # estimated from data
        "kernel": "rbf",
        "gamma": "auto"
    },
    "statistical_3sigma": {
        "threshold_multiplier": 3.0  # classic 3-sigma rule
    }
}

# Internal ablation baselines (all use same train/val/test split)
ABLATION_METHODS = [
    "simple_threshold",  # 3-sigma on raw features
    "lstm_only",  # forecast error threshold
    "ae_only",  # reconstruction error threshold
    "or_combination",  # either detector flags
    "and_combination",  # both detectors flag
    "confidence_max",  # max of normalized confidences
    "confidence_weighted",  # weighted combination
    "smart_hybrid",  # proposed fault-specific logic
]

# ============================================================================
# OUTPUT PATHS
# ============================================================================
RAW_DATA_PATH = "data/raw/simulated_spindle_data.csv"
PROCESSED_DATA_PATH = "data/processed/spindle_data_with_anomalies.csv"

LSTM_MODEL_PATH = "outputs/lstm_model.h5"
LSTM_SCALER_PATH = "outputs/lstm_scaler.npz"

AE_MODEL_PATH = "outputs/autoencoder_model.h5"
AE_META_PATH = "outputs/autoencoder_meta.npz"

DETECTION_RESULTS_PATH = "outputs/detection_results.csv"
ABLATION_RESULTS_PATH = "outputs/ablation_study.csv"
MULTIRUN_RESULTS_PATH = "outputs/multi_run_validation.csv"
STATISTICAL_TESTS_PATH = "outputs/statistical_tests.txt"

# ============================================================================
# DOCUMENTATION FLAGS
# ============================================================================
VERBOSE_LOGGING = True  # detailed console output during experiments
SAVE_INTERMEDIATE_RESULTS = True  # save per-run results for inspection
GENERATE_DIAGNOSTIC_PLOTS = False  # set True for visualization (not needed for publication)
