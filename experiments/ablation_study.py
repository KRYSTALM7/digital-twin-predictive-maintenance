"""
Rigorous ablation study comparing all detection methods on identical data.

This module ensures fair comparison by:
1. Using the same train/validation/test splits for all methods
2. Calibrating thresholds only on validation data (never test)
3. Reporting performance only on test data
4. Using identical feature representations where possible
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from experiments.experiment_config import *


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    }


class BaselineDetector:
    """Base class for all anomaly detectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.threshold = None
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Fit detector and calibrate threshold on validation data."""
        raise NotImplementedError
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict anomalies on test data."""
        raise NotImplementedError


class SimpleThresholdDetector(BaselineDetector):
    """Simple statistical threshold (3-sigma rule on raw features)."""
    
    def __init__(self, sigma_multiplier: float = 3.0):
        super().__init__("Simple 3-Sigma Threshold")
        self.sigma_multiplier = sigma_multiplier
        self.train_mean = None
        self.train_std = None
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        self.train_mean = X_train.mean(axis=0)
        self.train_std = X_train.std(axis=0)
        self.train_std[self.train_std == 0] = 1.0  # avoid division by zero
        
        # Threshold is the sigma multiplier (fixed, not tuned on validation)
        self.threshold = self.sigma_multiplier
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        # Compute max deviation across all features (Mahalanobis-like)
        deviations = np.abs(X_test - self.train_mean) / self.train_std
        max_deviations = deviations.max(axis=1)
        return (max_deviations > self.threshold).astype(int)


class IsolationForestDetector(BaselineDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self):
        super().__init__("Isolation Forest")
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # Estimate contamination from validation labels
        contamination = max(y_val.mean(), 0.01)  # at least 1%
        contamination = min(contamination, 0.5)  # at most 50%
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test_scaled = self.scaler.transform(X_test)
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_test_scaled)
        return (predictions == -1).astype(int)


class OneClassSVMDetector(BaselineDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self):
        super().__init__("One-Class SVM")
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # Estimate nu from validation labels
        nu = max(y_val.mean() * 2, 0.01)  # slightly more aggressive
        nu = min(nu, 0.5)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='auto'
        )
        self.model.fit(X_train_scaled)
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test_scaled = self.scaler.transform(X_test)
        # One-Class SVM returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_test_scaled)
        return (predictions == -1).astype(int)


class LSTMOnlyDetector(BaselineDetector):
    """LSTM forecast error detector."""
    
    def __init__(self, forecast_errors_train: np.ndarray, forecast_errors_val: np.ndarray,
                 sigma_multiplier: float = 2.0):
        super().__init__("LSTM Forecast Only")
        self.forecast_errors_train = forecast_errors_train
        self.forecast_errors_val = forecast_errors_val
        self.sigma_multiplier = sigma_multiplier
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # Calibrate threshold on training errors only
        train_mean = self.forecast_errors_train.mean()
        train_std = self.forecast_errors_train.std()
        self.threshold = train_mean + self.sigma_multiplier * train_std
    
    def predict(self, forecast_errors_test: np.ndarray) -> np.ndarray:
        return (forecast_errors_test > self.threshold).astype(int)


class AutoencoderOnlyDetector(BaselineDetector):
    """Autoencoder reconstruction error detector."""
    
    def __init__(self, ae_errors_train: np.ndarray, ae_errors_val: np.ndarray,
                 sigma_multiplier: float = 2.0):
        super().__init__("Autoencoder Only")
        self.ae_errors_train = ae_errors_train
        self.ae_errors_val = ae_errors_val
        self.sigma_multiplier = sigma_multiplier
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # Calibrate threshold on training errors only
        train_mean = self.ae_errors_train.mean()
        train_std = self.ae_errors_train.std()
        self.threshold = train_mean + self.sigma_multiplier * train_std
    
    def predict(self, ae_errors_test: np.ndarray) -> np.ndarray:
        return (ae_errors_test > self.threshold).astype(int)


class HybridORDetector(BaselineDetector):
    """Simple OR combination of LSTM and AE."""
    
    def __init__(self, lstm_detector: LSTMOnlyDetector, ae_detector: AutoencoderOnlyDetector):
        super().__init__("Hybrid OR")
        self.lstm_detector = lstm_detector
        self.ae_detector = ae_detector
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        # Individual detectors already fitted
        pass
    
    def predict(self, forecast_errors_test: np.ndarray, ae_errors_test: np.ndarray) -> np.ndarray:
        lstm_pred = self.lstm_detector.predict(forecast_errors_test)
        ae_pred = self.ae_detector.predict(ae_errors_test)
        return np.maximum(lstm_pred, ae_pred)


class HybridANDDetector(BaselineDetector):
    """Simple AND combination of LSTM and AE."""
    
    def __init__(self, lstm_detector: LSTMOnlyDetector, ae_detector: AutoencoderOnlyDetector):
        super().__init__("Hybrid AND")
        self.lstm_detector = lstm_detector
        self.ae_detector = ae_detector
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        pass
    
    def predict(self, forecast_errors_test: np.ndarray, ae_errors_test: np.ndarray) -> np.ndarray:
        lstm_pred = self.lstm_detector.predict(forecast_errors_test)
        ae_pred = self.ae_detector.predict(ae_errors_test)
        return np.minimum(lstm_pred, ae_pred)


def run_comprehensive_ablation_study(
    features: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,  # 'train', 'val', or 'test'
    forecast_errors: np.ndarray,
    ae_errors: np.ndarray,
    feature_confidences: Dict[str, np.ndarray],  # pre-computed confidence scores
    smart_hybrid_predictions: np.ndarray  # from main detection pipeline
) -> pd.DataFrame:
    """
    Run complete ablation study comparing all methods.
    
    All methods use:
    - Same train/val/test split
    - Thresholds calibrated on validation set only
    - Performance evaluated on test set only
    
    Returns:
        DataFrame with F1/Precision/Recall for each method
    """
    
    # Extract splits
    train_mask = (split == 'train')
    val_mask = (split == 'val')
    test_mask = (split == 'test')
    
    X_train = features[train_mask]
    X_val = features[val_mask]
    X_test = features[test_mask]
    
    y_val = labels[val_mask]
    y_test = labels[test_mask]
    
    # Forecast/AE errors are already windowed and aligned with post-window features
    forecast_train = forecast_errors[train_mask]
    forecast_val = forecast_errors[val_mask]
    forecast_test = forecast_errors[test_mask]
    
    ae_train = ae_errors[train_mask]
    ae_val = ae_errors[val_mask]
    ae_test = ae_errors[test_mask]
    
    results = []
    
    # ========================================================================
    # External Baselines
    # ========================================================================
    
    print("[Ablation] Running Simple 3-Sigma Threshold...")
    simple_detector = SimpleThresholdDetector(sigma_multiplier=3.0)
    simple_detector.fit(X_train, X_val, y_val)
    simple_pred = simple_detector.predict(X_test)
    simple_metrics = compute_binary_metrics(y_test, simple_pred)
    simple_metrics['method'] = 'Simple 3-Sigma'
    results.append(simple_metrics)
    
    print("[Ablation] Running Isolation Forest...")
    iso_detector = IsolationForestDetector()
    iso_detector.fit(X_train, X_val, y_val)
    iso_pred = iso_detector.predict(X_test)
    iso_metrics = compute_binary_metrics(y_test, iso_pred)
    iso_metrics['method'] = 'Isolation Forest'
    results.append(iso_metrics)
    
    print("[Ablation] Running One-Class SVM...")
    ocsvm_detector = OneClassSVMDetector()
    ocsvm_detector.fit(X_train, X_val, y_val)
    ocsvm_pred = ocsvm_detector.predict(X_test)
    ocsvm_metrics = compute_binary_metrics(y_test, ocsvm_pred)
    ocsvm_metrics['method'] = 'One-Class SVM'
    results.append(ocsvm_metrics)
    
    # ========================================================================
    # Internal Component Ablations
    # ========================================================================
    
    print("[Ablation] Running LSTM-Only...")
    lstm_only = LSTMOnlyDetector(forecast_train, forecast_val, sigma_multiplier=SIGMA_MULTIPLIER_LSTM)
    lstm_only.fit(None, None, y_val)
    lstm_pred = lstm_only.predict(forecast_test)
    lstm_metrics = compute_binary_metrics(y_test, lstm_pred)
    lstm_metrics['method'] = 'LSTM Only'
    results.append(lstm_metrics)
    
    print("[Ablation] Running Autoencoder-Only...")
    ae_only = AutoencoderOnlyDetector(ae_train, ae_val, sigma_multiplier=SIGMA_MULTIPLIER_AE)
    ae_only.fit(None, None, y_val)
    ae_pred = ae_only.predict(ae_test)
    ae_metrics = compute_binary_metrics(y_test, ae_pred)
    ae_metrics['method'] = 'Autoencoder Only'
    results.append(ae_metrics)
    
    print("[Ablation] Running Hybrid OR...")
    hybrid_or = HybridORDetector(lstm_only, ae_only)
    hybrid_or.fit(None, None, y_val)
    or_pred = hybrid_or.predict(forecast_test, ae_test)
    or_metrics = compute_binary_metrics(y_test, or_pred)
    or_metrics['method'] = 'Hybrid OR'
    results.append(or_metrics)
    
    print("[Ablation] Running Hybrid AND...")
    hybrid_and = HybridANDDetector(lstm_only, ae_only)
    hybrid_and.fit(None, None, y_val)
    and_pred = hybrid_and.predict(forecast_test, ae_test)
    and_metrics = compute_binary_metrics(y_test, and_pred)
    and_metrics['method'] = 'Hybrid AND'
    results.append(and_metrics)
    
    # ========================================================================
    # Proposed Method
    # ========================================================================
    
    print("[Ablation] Evaluating SMART Hybrid (Proposed)...")
    smart_pred_test = smart_hybrid_predictions[test_mask]
    smart_metrics = compute_binary_metrics(y_test, smart_pred_test)
    smart_metrics['method'] = 'SMART Hybrid (Proposed)'
    results.append(smart_metrics)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by F1 score descending
    df_results = df_results.sort_values('f1', ascending=False)
    
    # Save results
    df_results.to_csv(ABLATION_RESULTS_PATH, index=False)
    print(f"\n[Ablation] Results saved to: {ABLATION_RESULTS_PATH}")
    
    return df_results


if __name__ == "__main__":
    print("This module should be called from the main detection pipeline.")
    print("See: experiments/run_full_validation.py")
