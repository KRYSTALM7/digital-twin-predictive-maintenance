"""
Multi-run validation with statistical analysis.

Runs the complete pipeline multiple times with different random seeds
to estimate variance and compute confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List
import sys
import os

sys.path.insert(0, os.path.abspath('.'))

from experiments.experiment_config import *


def run_single_experiment(seed: int, results_dir: str = "outputs/runs") -> Dict:
    """
    Run complete pipeline with given seed.
    
    Returns dict with all metrics for this run.
    """
    print(f"\n{'='*80}")
    print(f"EXPERIMENTAL RUN: SEED {seed}")
    print(f"{'='*80}\n")
    
    # Create output directory for this run
    run_dir = os.path.join(results_dir, f"seed_{seed:02d}")
    os.makedirs(run_dir, exist_ok=True)
    
    # ========================================================================
    # Step 1: Simulation
    # ========================================================================
    print(f"[Run {seed}] Step 1/5: Running simulation...")
    from src.simulation.simpy_simulation import run_simulation
    
    sim_output = os.path.join(run_dir, "simulated_spindle_data.csv")
    run_simulation(
        duration=SIMULATION_DURATION,
        mean_time_to_failure=MEAN_TIME_TO_FAILURE,
        mean_repair_time=MEAN_REPAIR_TIME,
        preventive_maintenance_interval=PREVENTIVE_MAINTENANCE_INTERVAL,
        output_csv_path=sim_output
    )
    
    # ========================================================================
    # Step 2: Preprocessing with anomaly injection
    # ========================================================================
    print(f"[Run {seed}] Step 2/5: Preprocessing and anomaly injection...")
    from src.preprocessing.data_preprocessor import preprocess_data
    
    processed_output = os.path.join(run_dir, "spindle_data_with_anomalies.csv")
    preprocess_data(
        raw_csv_path=sim_output,
        processed_csv_path=processed_output,
        anomaly_method="physics",
        train_fraction=TRAIN_FRACTION,
        seed=seed  # Different anomaly patterns per run
    )
    
    # ========================================================================
    # Step 3: Train LSTM
    # ========================================================================
    print(f"[Run {seed}] Step 3/5: Training LSTM forecaster...")
    from src.models.lstm_forecast import train_lstm
    
    lstm_model_path = os.path.join(run_dir, "lstm_model.h5")
    lstm_scaler_path = os.path.join(run_dir, "lstm_scaler.npz")
    
    train_lstm(
        processed_csv_path=processed_output,
        model_output_path=lstm_model_path,
        scaler_output_path=lstm_scaler_path,
        window_size=LSTM_WINDOW_SIZE,
        epochs=LSTM_EPOCHS
    )
    
    # ========================================================================
    # Step 4: Train Autoencoder
    # ========================================================================
    print(f"[Run {seed}] Step 4/5: Training autoencoder...")
    from src.models.autoencoder import train_autoencoder
    
    ae_model_path = os.path.join(run_dir, "autoencoder_model.h5")
    ae_meta_path = os.path.join(run_dir, "autoencoder_meta.npz")
    
    train_autoencoder(
        processed_csv_path=processed_output,
        model_output_path=ae_model_path,
        meta_output_path=ae_meta_path,
        epochs=AE_EPOCHS,
        batch_size=AE_BATCH_SIZE
    )
    
    # ========================================================================
    # Step 5: Detection and metrics
    # ========================================================================
    print(f"[Run {seed}] Step 5/5: Running detection and computing metrics...")
    from src.feedback.detection_and_metrics import run_detection_and_metrics
    
    results_csv = os.path.join(run_dir, "detection_results.csv")
    
    metrics = run_detection_and_metrics(
        processed_csv_path=processed_output,
        out_csv=results_csv,
        forecast_threshold_k=SIGMA_MULTIPLIER_LSTM
    )
    
    # Extract key metrics for this run
    result = {
        'seed': seed,
        'smart_f1': metrics['combined_metrics']['f1'],
        'smart_precision': metrics['combined_metrics']['precision'],
        'smart_recall': metrics['combined_metrics']['recall'],
        'smart_accuracy': metrics['combined_metrics']['accuracy'],
        'lstm_f1': metrics['forecast_only_metrics']['f1'],
        'lstm_precision': metrics['forecast_only_metrics']['precision'],
        'lstm_recall': metrics['forecast_only_metrics']['recall'],
        'ae_f1': metrics['ae_only_metrics']['f1'],
        'ae_precision': metrics['ae_only_metrics']['precision'],
        'ae_recall': metrics['ae_only_metrics']['recall'],
        'temp_f1': metrics['temp_combined_metrics']['f1'],
        'temp_recall': metrics['temp_combined_metrics']['recall'],
        'vib_f1': metrics['vib_combined_metrics']['f1'],
        'vib_recall': metrics['vib_combined_metrics']['recall'],
        'mttd_vib': metrics.get('mttd_vib', None),
        'mttd_temp': metrics.get('mttd_temp', None),
    }
    
    return result


def compute_statistics(results: List[Dict]) -> pd.DataFrame:
    """Compute mean, std, CI, and min/max across runs."""
    df = pd.DataFrame(results)
    
    stats_list = []
    
    # Metrics to analyze
    metrics = [
        'smart_f1', 'smart_precision', 'smart_recall', 'smart_accuracy',
        'lstm_f1', 'lstm_precision', 'lstm_recall',
        'ae_f1', 'ae_precision', 'ae_recall',
        'temp_f1', 'temp_recall',
        'vib_f1', 'vib_recall',
        'mttd_vib', 'mttd_temp'
    ]
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].dropna().values
        
        if len(values) == 0:
            continue
        
        mean = values.mean()
        std = values.std()
        
        # 95% confidence interval using t-distribution
        if len(values) > 1:
            ci_low, ci_high = stats.t.interval(
                0.95,
                len(values) - 1,
                loc=mean,
                scale=stats.sem(values)
            )
        else:
            ci_low = ci_high = mean
        
        stats_list.append({
            'metric': metric,
            'mean': mean,
            'std': std,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'min': values.min(),
            'max': values.max(),
            'n_runs': len(values)
        })
    
    return pd.DataFrame(stats_list)


def perform_statistical_tests(results: List[Dict]) -> Dict:
    """Perform paired t-tests comparing methods."""
    df = pd.DataFrame(results)
    
    tests = {}
    
    # Test 1: SMART vs LSTM-only
    if 'smart_f1' in df.columns and 'lstm_f1' in df.columns:
        t_stat, p_val = stats.ttest_rel(df['smart_f1'], df['lstm_f1'])
        tests['smart_vs_lstm'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < ALPHA,
            'smart_better': df['smart_f1'].mean() > df['lstm_f1'].mean()
        }
    
    # Test 2: SMART vs AE-only
    if 'smart_f1' in df.columns and 'ae_f1' in df.columns:
        t_stat, p_val = stats.ttest_rel(df['smart_f1'], df['ae_f1'])
        tests['smart_vs_ae'] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < ALPHA,
            'smart_better': df['smart_f1'].mean() > df['ae_f1'].mean()
        }
    
    return tests


def run_multi_run_validation(n_runs: int = N_EXPERIMENTAL_RUNS, 
                             seeds: List[int] = None) -> pd.DataFrame:
    """
    Run complete validation with multiple seeds.
    
    Args:
        n_runs: Number of independent runs
        seeds: List of random seeds (if None, uses range(n_runs))
    
    Returns:
        DataFrame with statistics across runs
    """
    if seeds is None:
        seeds = list(range(n_runs))
    
    results = []
    failed_runs = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'#'*80}")
        print(f"# STARTING RUN {i+1}/{n_runs} (Seed: {seed})")
        print(f"{'#'*80}")
        
        try:
            result = run_single_experiment(seed)
            results.append(result)
            
            print(f"\n[Run {i+1}] SMART F1: {result['smart_f1']:.4f}")
            
        except Exception as e:
            print(f"\n[ERROR] Run {i+1} (seed {seed}) failed: {e}")
            failed_runs.append(seed)
            continue
    
    if len(results) == 0:
        raise RuntimeError("All experimental runs failed!")
    
    # Compute statistics
    print(f"\n{'='*80}")
    print("MULTI-RUN VALIDATION RESULTS")
    print(f"{'='*80}\n")
    print(f"Successfully completed: {len(results)}/{n_runs} runs")
    if failed_runs:
        print(f"Failed runs (seeds): {failed_runs}")
    
    # Individual run results
    df_runs = pd.DataFrame(results)
    df_runs.to_csv(MULTIRUN_RESULTS_PATH, index=False)
    print(f"\nPer-run results saved to: {MULTIRUN_RESULTS_PATH}")
    
    # Statistics across runs
    df_stats = compute_statistics(results)
    
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(df_stats.to_string(index=False))
    
    # Statistical significance tests
    tests = perform_statistical_tests(results)
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    test_output = []
    
    for test_name, test_result in tests.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print(f"  t-statistic: {test_result['t_statistic']:.4f}")
        print(f"  p-value:     {test_result['p_value']:.6f}")
        print(f"  Significant (α={ALPHA}): {'YES' if test_result['significant'] else 'NO'}")
        
        if test_result['smart_better']:
            print(f"  Result: SMART hybrid performs better")
        else:
            print(f"  Result: Baseline performs better or equal")
        
        test_output.append(f"{test_name}: t={test_result['t_statistic']:.4f}, "
                          f"p={test_result['p_value']:.6f}, "
                          f"sig={'YES' if test_result['significant'] else 'NO'}")
    
    # Save statistical tests
    with open(STATISTICAL_TESTS_PATH, 'w') as f:
        f.write("STATISTICAL SIGNIFICANCE TESTS\n")
        f.write("="*80 + "\n\n")
        for line in test_output:
            f.write(line + "\n")
    
    print(f"\nStatistical tests saved to: {STATISTICAL_TESTS_PATH}")
    
    return df_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run multi-run validation')
    parser.add_argument('--n_runs', type=int, default=N_EXPERIMENTAL_RUNS,
                       help=f'Number of runs (default: {N_EXPERIMENTAL_RUNS})')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 3 runs only')
    
    args = parser.parse_args()
    
    n_runs = 3 if args.quick else args.n_runs
    
    print(f"\nStarting multi-run validation with {n_runs} runs...")
    print(f"This will take approximately {n_runs * 5} minutes.\n")
    
    df_stats = run_multi_run_validation(n_runs=n_runs)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"\nKey Result:")
    smart_row = df_stats[df_stats['metric'] == 'smart_f1'].iloc[0]
    print(f"SMART Hybrid F1: {smart_row['mean']:.4f} ± {smart_row['std']:.4f}")
    print(f"95% CI: [{smart_row['ci_95_low']:.4f}, {smart_row['ci_95_high']:.4f}]")
