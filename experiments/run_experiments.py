"""
Publication-ready experiment runner.

Provides convenient functions for running:
1. Sanity test (quick check with default thresholds)
2. Threshold calibration (grid search on validation set)
3. Ablation study (compare all methods fairly)
4. Multi-run validation (estimate variance across seeds)
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.preprocessing.data_preprocessor import preprocess_data
from src.feedback.detection_and_metrics import run_detection_and_metrics
from experiments.multi_run_validation import run_multi_run_validation
from experiments.experiment_config import *


def sanity_test(seed: int = 42):
    """
    Quick sanity check with default parameters.
    
    - Uses default thresholds (no calibration)
    - No ablation study
    - Single seed
    - Fast execution (~1-2 minutes)
    
    Purpose: Verify pipeline runs without errors
    """
    print("="*70)
    print("SANITY TEST")
    print("="*70)
    print(f"Seed: {seed}")
    print("Mode: Standard (no calibration, no ablation)")
    print("="*70 + "\n")
    
    # Preprocess with specified seed
    print("[1/2] Preprocessing data...")
    preprocess_data(seed=seed)
    
    # Run detection with default thresholds
    print("\n[2/2] Running detection...")
    results = run_detection_and_metrics(
        calibrate_thresholds=False,
        run_ablation=False
    )
    
    print("\n" + "="*70)
    print("SANITY TEST COMPLETE")
    print("="*70)
    print(f"F1: {results['combined_metrics']['f1']:.4f}")
    print(f"Precision: {results['combined_metrics']['precision']:.4f}")
    print(f"Recall: {results['combined_metrics']['recall']:.4f}")
    print("="*70)
    
    return results


def run_ablation_only(seed: int = 42):
    """
    Run ablation study with threshold calibration.
    
    - Calibrates thresholds on validation set
    - Compares SMART against 7 baselines
    - All methods use identical train/val/test splits
    - Single seed
    - Execution time: ~5-10 minutes
    
    Purpose: Fair comparison of all detection methods
    """
    print("="*70)
    print("ABLATION STUDY")
    print("="*70)
    print(f"Seed: {seed}")
    print("Mode: Calibration + Ablation")
    print("Baselines: 3-Sigma, Isolation Forest, One-Class SVM, LSTM, AE, OR, AND, SMART")
    print("="*70 + "\n")
    
    # Preprocess with specified seed
    print("[1/2] Preprocessing data...")
    preprocess_data(seed=seed)
    
    # Run detection with calibration and ablation
    print("\n[2/2] Running detection with ablation...")
    results = run_detection_and_metrics(
        calibrate_thresholds=True,
        run_ablation=True,
        validation_fraction=VAL_FRACTION
    )
    
    print("\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    print(f"Results saved to: {ABLATION_RESULTS_PATH}")
    print(f"SMART F1 (test): {results['combined_metrics']['f1']:.4f}")
    if results['ablation_results'] is not None:
        print("\nTop 3 methods by F1:")
        print(results['ablation_results'][['method', 'f1', 'precision', 'recall']].head(3).to_string(index=False))
    print("="*70)
    
    return results


def run_threshold_calibration_only(seed: int = 42, custom_ranges: dict = None):
    """
    Run threshold calibration without ablation study.
    
    - Grid search over threshold combinations on validation set
    - Reports calibrated thresholds
    - Evaluates calibrated model on test set
    - Single seed
    - Execution time: ~2-3 minutes
    
    Purpose: Find optimal thresholds for SMART detector
    
    Args:
        seed: Random seed for preprocessing
        custom_ranges: Optional dict with custom threshold search ranges
                      {'strong': (min, max, step), 'moderate': (min, max, step), 'weak': (min, max, step)}
    """
    print("="*70)
    print("THRESHOLD CALIBRATION")
    print("="*70)
    print(f"Seed: {seed}")
    print("Mode: Calibration only (no ablation)")
    if custom_ranges:
        print(f"Custom ranges: {custom_ranges}")
    else:
        print("Using default search ranges from experiment_config.py")
    print("="*70 + "\n")
    
    # Preprocess with specified seed
    print("[1/2] Preprocessing data...")
    preprocess_data(seed=seed)
    
    # Run detection with calibration only
    print("\n[2/2] Calibrating thresholds...")
    results = run_detection_and_metrics(
        calibrate_thresholds=True,
        run_ablation=False,
        validation_fraction=VAL_FRACTION,
        threshold_ranges=custom_ranges
    )
    
    print("\n" + "="*70)
    print("THRESHOLD CALIBRATION COMPLETE")
    print("="*70)
    if results['calibrated_thresholds']:
        print(f"Calibrated thresholds: {results['calibrated_thresholds']}")
    print(f"Test F1: {results['combined_metrics']['f1']:.4f}")
    print(f"Test Precision: {results['combined_metrics']['precision']:.4f}")
    print(f"Test Recall: {results['combined_metrics']['recall']:.4f}")
    print("="*70)
    
    return results


def run_quick_multirun_validation(n_runs: int = 3):
    """
    Quick multi-run validation with few seeds.
    
    - Runs full pipeline (calibration + ablation) for each seed
    - Computes mean and std of metrics across runs
    - Execution time: ~15-30 minutes for 3 runs
    
    Purpose: Quick estimate of result stability
    
    Args:
        n_runs: Number of independent runs (default: 3)
    """
    print("="*70)
    print(f"QUICK MULTI-RUN VALIDATION ({n_runs} runs)")
    print("="*70)
    print("Each run: Preprocess → Calibrate → Ablation → Evaluate")
    print(f"Seeds: {list(range(n_runs))}")
    print("="*70 + "\n")
    
    results = run_multi_run_validation(n_runs=n_runs)
    
    print("\n" + "="*70)
    print(f"MULTI-RUN VALIDATION COMPLETE ({n_runs} runs)")
    print("="*70)
    print(f"Results saved to: {MULTIRUN_RESULTS_PATH}")
    print("\nMethod performance (mean ± std):")
    print(results['summary'].to_string(index=False))
    print("="*70)
    
    return results


def run_full_multirun_validation(n_runs: int = 10):
    """
    Full multi-run validation for publication.
    
    - Runs full pipeline for 10 independent seeds
    - Computes confidence intervals
    - Performs statistical significance tests
    - Execution time: ~1-2 hours for 10 runs
    
    Purpose: Publication-quality variance estimation and statistical testing
    
    Args:
        n_runs: Number of independent runs (default: 10)
    """
    print("="*70)
    print(f"FULL MULTI-RUN VALIDATION ({n_runs} runs)")
    print("="*70)
    print("Each run: Preprocess → Train → Calibrate → Ablation → Evaluate")
    print(f"Seeds: {list(range(n_runs))}")
    print("This will take approximately 1-2 hours.")
    print("="*70 + "\n")
    
    results = run_multi_run_validation(n_runs=n_runs)
    
    print("\n" + "="*70)
    print(f"FULL MULTI-RUN VALIDATION COMPLETE ({n_runs} runs)")
    print("="*70)
    print(f"Results saved to: {MULTIRUN_RESULTS_PATH}")
    print(f"Statistical tests saved to: {STATISTICAL_TESTS_PATH}")
    print("\nMethod performance (mean ± std):")
    print(results['summary'].to_string(index=False))
    print("\nStatistical significance tests:")
    print(results['statistical_tests'])
    print("="*70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run digital twin experiments")
    parser.add_argument("mode", choices=["sanity", "ablation", "calibration", "quick", "full"],
                        help="Experiment mode to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (for single-run experiments)")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Number of runs (for multi-run experiments)")
    
    args = parser.parse_args()
    
    if args.mode == "sanity":
        sanity_test(seed=args.seed)
    elif args.mode == "ablation":
        run_ablation_only(seed=args.seed)
    elif args.mode == "calibration":
        run_threshold_calibration_only(seed=args.seed)
    elif args.mode == "quick":
        n_runs = args.n_runs if args.n_runs else 3
        run_quick_multirun_validation(n_runs=n_runs)
    elif args.mode == "full":
        n_runs = args.n_runs if args.n_runs else 10
        run_full_multirun_validation(n_runs=n_runs)
