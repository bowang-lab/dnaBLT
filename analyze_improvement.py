import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from math import log, exp
from scipy.stats import t

def load_wandb_csv(file_path: str) -> pd.DataFrame:
    """Load a wandb export CSV file."""
    return pd.read_csv(file_path)

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


class CITracker:
    def __init__(self, idx2, C_fixed, alpha=0.05):
        self.idx2 = idx2
        self.C    = C_fixed
        self.alpha = alpha
        self.n = self.Sx = self.Sy = self.Sxx = self.Sxy = 0
        self.SSE = 0      # sum of squared residuals

    def update(self, step, loss):
        if step <= 0:
            raise ValueError("step must be > 0")

        x = log(step)
        y = log(loss - self.C)

        # -------- accumulate moments --------
        self.n  += 1
        self.Sx += x
        self.Sy += y
        self.Sxx += x*x
        self.Sxy += x*y

        # need ≥3 distinct points before we can form σ²
        if self.n < 10:
            return None, float("inf")

        mean_x       = self.Sx / self.n
        Sxx_central  = self.Sxx - self.Sx**2 / self.n          # ∑(x_i-mean_x)^2
        if Sxx_central <= 1e-14:                               # duplicated steps?
            return None, float("inf")

        # ------- OLS coefficients (closed form) -------
        beta  = (self.Sxy - self.Sx*self.Sy / self.n) / Sxx_central
        logA  = (self.Sy - beta*self.Sx) / self.n

        # running residual sum of squares
        resid  = y - (logA + beta * x)
        self.SSE += resid**2
        sigma2 = self.SSE / (self.n - 2)

        # ------- CI half-width at x2 -------
        x2      = log(self.idx2)
        var_pred = sigma2 * (1/self.n + (x2 - mean_x)**2 / Sxx_central)
        se_pred  = var_pred**0.5                                  # always real-valued

        t_mult   = t.ppf(1 - self.alpha/2, df=self.n - 2)
        half_w   = t_mult * se_pred
        pred     = exp(logA + beta * x2) + self.C
        rel_hw   = half_w / pred
        return pred, rel_hw

def analyze_two_points(df: pd.DataFrame, loss_column: str, step_column: str, idx2: int, window_size=11, min_points_skip=0):
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
    if not (0 <= idx2 < len(df)):
        print(f"Invalid indices. Ensure 0 <= idx1 < idx2 < {len(df)}.")
        return
        
    # Adjust indices to account for skipped points
    start_idx = min_points_skip
    
    # Get data with initial points skipped
    df_analysis = df.iloc[start_idx:].reset_index(drop=True)
    idx2_adj = idx2 - start_idx
    
    # --- 1. Load and Smooth Data --- 
    all_steps = df_analysis[step_column].values.astype(float)
    all_raw_losses = df_analysis[loss_column].values
    all_smoothed_losses = smooth_data(all_raw_losses, window_size)
    
    # Get values at the specified indices (adjusted for skipped points)
    step_val_at_idx2 = all_steps[idx2_adj]
    raw_loss_at_idx2 = all_raw_losses[idx2_adj]
    smoothed_loss_at_idx2 = all_smoothed_losses[idx2_adj]

    tracker = CITracker(idx2=idx2_adj, C_fixed=1.19) # irreducible loss

    idx1 = None
    for step, loss in zip(all_steps, all_smoothed_losses):
        pred, rel_hw = tracker.update(step, loss)
        if step > all_steps[-1] * 0.1 and rel_hw <= 0.021:
            idx1 = np.where(all_steps == step)[0][0] + start_idx
            print(idx1 / idx2)
            break
    
    if idx1 is None:
        print("No suitable idx1 found where relative half-width <= 0.021 after skipping initial points.")
        return

    idx1_adj = idx1 - start_idx
    step_val_at_idx1 = all_steps[idx1_adj]
    raw_loss_at_idx1 = all_raw_losses[idx1_adj]
    smoothed_loss_at_idx1 = all_smoothed_losses[idx1_adj]
    
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
    
    # idx1, idx2 = 189, 378
    # idx1, idx2 = 193, 378
    idx2 = 219

    analyze_two_points(df, loss_column, step_column, idx2)

def main():
    # file_path = "/Users/arnavshah/Code/dnaBLT/run_curves/wandb_export_2025-05-28T17_51_03.274-04_00.csv"
    file_path = "/Users/arnavshah/Code/dnaBLT/run_curves/wandb_export_2025-05-28T18_09_04.251-04_00.csv"
    if os.path.exists(file_path):
        analyze_file(file_path)
    else:
        print(f"File not found: {file_path}")
        return

if __name__ == "__main__":
    main()
