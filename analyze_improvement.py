import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from scipy.optimize import curve_fit

def load_wandb_csv(file_path: str) -> pd.DataFrame:
    """Load a wandb export CSV file."""
    return pd.read_csv(file_path)

def calculate_improvement_rates(df: pd.DataFrame, 
                              loss_column: str = 'val/loss', 
                              step_column: str = '_step',
                              window_size: int = 5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the improvement rates using the specified formula:
    recent_ratio = (loss[-1] - loss[-2]) / loss[-2]   # fractional drop
    slope = recent_ratio / (logC[-1] - logC[-2])  # per decade
    
    Args:
        df: DataFrame containing the training data
        loss_column: Name of the column containing loss values
        step_column: Name of the column containing step values
        window_size: Number of steps to consider for smoothing
        
    Returns:
        Tuple of (fractional_drops, slopes_per_decade, should_stop)
    """
    # Sort by step to ensure correct ordering
    df = df.sort_values(by=step_column).copy()
    
    # Calculate fractional drop in loss
    loss = df[loss_column].values
    fractional_drops = np.zeros_like(loss, dtype=float)
    fractional_drops[1:] = (loss[1:] - loss[:-1]) / loss[:-1]
    
    # Calculate steps (convert to log scale for per-decade calculation)
    steps = df[step_column].values
    log_steps = np.log10(np.maximum(steps, 1))  # Avoid log(0)
    
    # Calculate slope per decade (change in fractional drop per log10 step)
    slopes_per_decade = np.zeros_like(steps, dtype=float)
    for i in range(2, len(steps)):
        if log_steps[i] != log_steps[i-1]:  # Avoid division by zero
            slopes_per_decade[i] = fractional_drops[i] / (log_steps[i] - log_steps[i-1])
    
    # Apply rolling mean to smooth the slopes
    smoothed_slopes = pd.Series(slopes_per_decade).rolling(
        window=window_size, min_periods=1).mean().values
    
    # Determine if training should stop (slope magnitude < 1%)
    should_stop = np.abs(smoothed_slopes) < 0.01
    
    return pd.Series(fractional_drops, index=df.index), \
           pd.Series(smoothed_slopes, index=df.index), \
           pd.Series(should_stop, index=df.index)

def plot_improvement_rates(df: pd.DataFrame, 
                         loss_column: str,
                         step_column: str,
                         fractional_drops: pd.Series,
                         slopes_per_decade: pd.Series,
                         should_stop: pd.Series,
                         title: str = 'Training Analysis',
                         output_path: Optional[str] = None):
    """
    Plot the loss curve, fractional drops, and improvement slopes.
    
    Args:
        df: DataFrame containing the training data
        loss_column: Name of the column containing loss values
        step_column: Name of the column containing step values
        fractional_drops: Fractional drop in loss at each step
        slopes_per_decade: Smoothed slopes of improvement per decade
        should_stop: Boolean series indicating if training should stop at each step
        title: Plot title
        output_path: Optional path to save the plot
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    steps = df[step_column]
    
    # Plot 1: Loss curve
    ax1.plot(steps, df[loss_column], 'b-', label='Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    # Mark points where training should stop
    stop_steps = steps[should_stop & (steps > steps.min())]
    if not stop_steps.empty:
        ax1.scatter(stop_steps, df.loc[stop_steps.index, loss_column], 
                   color='red', s=50, marker='x', label='Should stop')
    ax1.legend()
    
    # Plot 2: Fractional drops
    ax2.plot(steps, fractional_drops, 'g-', label='Fractional Drop')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_ylabel('(L(t) - L(t-1)) / L(t-1)')
    ax2.set_title('Fractional Drop in Loss')
    ax2.grid(True)
    
    # Plot 3: Slopes per decade
    ax3.plot(steps, slopes_per_decade, 'r-', label='Slope per Decade')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, label='1% Threshold')
    ax3.axhline(y=-0.01, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Slope (per decade)')
    ax3.set_title('Improvement Rate (Slope per Decade)')
    ax3.grid(True)
    ax3.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def smooth_data(y, window_size=5):
    """
    Apply a running average smoother to the data with proper edge handling.
    
    Args:
        y: Input data to be smoothed
        window_size: Size of the moving window (must be odd)
        
    Returns:
        Smoothed data with the same length as input
    """
    if window_size < 2:
        return y
        
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
        
    half_window = window_size // 2
    y_padded = np.pad(y, (half_window, half_window), mode='edge')
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(y_padded, window, mode='valid')
    
    # Ensure output has the same length as input
    if len(smoothed) > len(y):
        diff = len(smoothed) - len(y)
        start = diff // 2
        smoothed = smoothed[start:start + len(y)]
    
    return smoothed

def analyze_two_points(df: pd.DataFrame, loss_column: str, step_column: str, idx1: int, idx2: int, window_size=11, min_points_skip=0):
    """
    Analyze improvement, fit power law up to idx1, predict at idx2.
    Skips initial min_points_skip points to avoid initialization effects.
    
    Args:
        df: DataFrame containing the data
        loss_column: Name of the column containing loss values
        step_column: Name of the column containing step values
        idx1: Index up to which to fit the power law
        idx2: Index at which to make prediction
        window_size: Size of the smoothing window
        min_points_skip: Number of initial points to skip (to avoid initialization instability)
    """
    if not (0 <= idx1 < len(df) and 0 <= idx2 < len(df) and idx1 < idx2):
        print(f"Invalid indices. Ensure 0 <= idx1 < idx2 < {len(df)}.")
        return
        
    # Adjust indices to account for skipped points
    min_points_skip = min(min_points_skip, idx1 - 1)  # Need at least 2 points after skipping
    start_idx = min_points_skip
    
    # Get data with initial points skipped
    df_analysis = df.iloc[start_idx:].reset_index(drop=True)
    idx1_adj = idx1 - start_idx
    idx2_adj = idx2 - start_idx
    
    # Ensure adjusted indices are valid
    if idx1_adj < 0 or idx2_adj >= len(df_analysis) or idx1_adj >= idx2_adj:
        print("Not enough data points after skipping initial points.")
        return
    
    # --- 1. Load and Smooth Data --- 
    all_steps = df_analysis[step_column].values.astype(float)
    all_raw_losses = df_analysis[loss_column].values
    all_smoothed_losses = smooth_data(all_raw_losses, window_size)
    
    # Get values at the specified indices (adjusted for skipped points)
    step_val_at_idx1 = all_steps[idx1_adj]
    step_val_at_idx2 = all_steps[idx2_adj]
    raw_loss_at_idx1 = all_raw_losses[idx1_adj]
    raw_loss_at_idx2 = all_raw_losses[idx2_adj]
    smoothed_loss_at_idx1 = all_smoothed_losses[idx1_adj]
    smoothed_loss_at_idx2 = all_smoothed_losses[idx2_adj]
    
    # Calculate slope based on smoothed data between idx1 and idx2
    log_steps_diff = np.log10(step_val_at_idx2) - np.log10(step_val_at_idx1) \
        if step_val_at_idx1 > 0 and step_val_at_idx2 > 0 else 0
    
    slope_percent = float('inf')
    if log_steps_diff != 0 and smoothed_loss_at_idx1 != 0:
        slope_decay = ((smoothed_loss_at_idx2 - smoothed_loss_at_idx1) / smoothed_loss_at_idx1) / log_steps_diff
        slope_percent = slope_decay * 100

    print(f"Analyzing from step {step_val_at_idx1} (idx1={idx1}, adj_idx={idx1_adj}) to step {step_val_at_idx2} (idx2={idx2}, adj_idx={idx2_adj}):")
    print(f"  Raw Loss:      {raw_loss_at_idx1:.6f} -> {raw_loss_at_idx2:.6f}")
    print(f"  Smoothed Loss: {smoothed_loss_at_idx1:.6f} -> {smoothed_loss_at_idx2:.6f} | Slope: {slope_percent:.2f}%/decade")

    # --- 2. Data for Fitting (up to idx1_adj) ---
    steps_for_fitting = all_steps[:idx1_adj + 1]
    losses_for_fitting = all_smoothed_losses[:idx1_adj + 1]
    
    # Only use points with positive steps for fitting
    fit_mask = (steps_for_fitting > 0)
    if not np.any(fit_mask):
        print("No valid steps > 0 for power law fitting up to idx1.")
        return
    
    steps_fit_masked = steps_for_fitting[fit_mask]
    losses_fit_masked = losses_for_fitting[fit_mask]

    if len(steps_fit_masked) < 3: # Need at least 3 points for 3 parameters
        print(f"Not enough data points ({len(steps_fit_masked)}) for fitting up to idx1. Need at least 3.")
        return

    # Define the power law function: L(t) = A * t^(-gamma) + C
    def power_law(t, A, gamma, C):
        return A * (t**-gamma) + C

    try:
        # Initial parameter guesses for fitting
        p0 = [losses_fit_masked[0], 0.5, losses_fit_masked[-1]]
        bounds = ([0, 0, 0], [np.inf, 5, losses_fit_masked[-1] if losses_fit_masked[-1] > 0 else np.inf])

        popt, _ = curve_fit(
            power_law,
            steps_fit_masked,
            losses_fit_masked,
            p0=p0,
            bounds=bounds,
            maxfev=400*len(steps_fit_masked) # Increased max iterations
        )

        # --- 3. Prediction (at step corresponding to idx2) --- 
        predicted_loss_val = power_law(step_val_at_idx2, *popt)
        # --- 4. Accuracy Measurement --- 
        error_val = predicted_loss_val - smoothed_loss_at_idx2
        relative_error_val = error_val / smoothed_loss_at_idx2 if smoothed_loss_at_idx2 != 0 else float('inf')
        
        print(f"\nPower Law Fit (based on data up to step {step_val_at_idx1}): L(t) = {popt[0]:.2e} * t^(-{popt[1]:.3f}) + {popt[2]:.4f}")
        print(f"Prediction for step {step_val_at_idx2}:")
        print(f"  Predicted Loss: {predicted_loss_val:.6f}")
        print(f"  Actual Smoothed Loss: {smoothed_loss_at_idx2:.6f}")
        print(f"  Error: {error_val:.2e} ({relative_error_val*100:.1f}%)")

        # --- 5. Visualization --- 
        plt.figure(figsize=(12, 6))

        # Plot raw data (all available points after skipping)
        plt.scatter(all_steps, all_smoothed_losses, color='lightgray', alpha=0.4, label='Raw Data')
        
        # Plot smoothed data (all available points after skipping)
        plt.plot(all_steps, all_smoothed_losses, 'b-', linewidth=2, label='Smoothed Data')

        # Plot power law curve (fitted up to idx1, extrapolated to idx2)
        # Ensure t_plot starts from the first positive step used in fitting and extends to step_to_predict_on
        t_plot_start = np.min(steps_fit_masked[steps_fit_masked > 0]) if np.any(steps_fit_masked > 0) else 1
        t_plot_end = step_val_at_idx2
        
        # Add vertical lines for key points
        plt.axvline(x=step_val_at_idx1, color='purple', linestyle='--', 
                   label=f'Fitting end (idx1={idx1}, step={step_val_at_idx1})')
        plt.axvline(x=step_val_at_idx2, color='red', linestyle='--', 
                   label=f'Prediction point (idx2={idx2}, step={step_val_at_idx2})')
        
        # Add a note about skipped points
        plt.annotate(f'Skipped first {min_points_skip} points', 
                    xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if t_plot_start < t_plot_end and step_val_at_idx2 > 0:
            t_for_curve_plot = np.linspace(t_plot_start, t_plot_end, 500)
            fitted_curve_on_plot = power_law(t_for_curve_plot, *popt)
            plt.plot(t_for_curve_plot, fitted_curve_on_plot, 'r--', linewidth=2, 
                     label=f'Power Law Fit (to {step_val_at_idx1}, pred to {step_val_at_idx2})')
        
        # Highlight the actual smoothed point at idx2
        if step_val_at_idx2 > 0:
            plt.scatter([step_val_at_idx2], [smoothed_loss_at_idx2], 
                        color='green', marker='o', s=100, zorder=10, 
                        label=f'Actual Smoothed at {step_val_at_idx2}: {smoothed_loss_at_idx2:.6f}')
            # Highlight the predicted point at idx2
            plt.scatter([step_val_at_idx2], [predicted_loss_val], 
                        color='orange', marker='x', s=100, zorder=11, 
                        label=f'Predicted at {step_val_at_idx2}: {predicted_loss_val:.6f}')

        # plt.xscale('log') # User had this commented, keeping it so
        plt.xlabel('Step') # Changed from 'Step (log scale)' as xscale is commented
        plt.ylabel('Loss')
        plt.title(f'Power Law Extrapolation: Fit up to Step {step_val_at_idx1}, Predict at Step {step_val_at_idx2}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()

    except RuntimeError as e:
        print(f"\nPower law fitting failed: {str(e)}")
        print("This often happens if data is noisy, C bound is too restrictive, or p0 is far off.")
        print(f"  Consider adjusting window_size for smoothing, bounds, or initial guesses (p0).")
        if 'popt' in locals():
             print(f"  Last attempted parameters: A={popt[0]:.2e}, gamma={popt[1]:.3f}, C={popt[2]:.4f}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_file(file_path: str):
    """Analyze a single wandb export file."""
    # Load data
    df = load_wandb_csv(file_path)
    
    # Find relevant columns (handle different possible column names)
    possible_loss_columns = ['restful-jazz-47 - val_entropy_loss', 'restful-flower-48 - val_entropy_loss']
    loss_column = next((col for col in possible_loss_columns if col in df.columns), None)
    
    if loss_column is None:
        print(f"Could not find loss column in {file_path}. Available columns: {df.columns.tolist()}")
        return
    
    step_column = '_step' if '_step' in df.columns else 'Step' if 'Step' in df.columns else df.columns[0]
    
    # Sort by step to ensure correct ordering
    df = df.sort_values(by=step_column)
    
    idx1, idx2 = 236, 378
    analyze_two_points(df, loss_column, step_column, idx1, idx2)

def main():
    file_path = "/Users/arnavshah/Code/dnaBLT/run_curves/wandb_export_2025-05-28T17_51_03.274-04_00.csv"
    if os.path.exists(file_path):
        analyze_file(file_path)
    else:
        print(f"File not found: {file_path}")
        return

if __name__ == "__main__":
    main()
