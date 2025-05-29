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
    """Apply a simple moving average smoother to the data."""
    if window_size < 2:
        return y
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode='same')

def analyze_two_points(df: pd.DataFrame, loss_column: str, step_column: str, idx1: int, idx2: int):
    """Analyze the improvement between two points in the training data."""
    if idx1 < 0 or idx2 >= len(df) or idx1 >= idx2:
        print("Invalid indices. Please ensure 0 <= idx1 < idx2 < len(df)")
        return
    
    # Get the values
    step1 = int(df[step_column].iloc[idx1])
    step2 = int(df[step_column].iloc[idx2])
    loss1 = df[loss_column].iloc[idx1]
    loss2 = df[loss_column].iloc[idx2]
    
    # Calculate metrics
    log_steps_diff = np.log10(step2) - np.log10(step1) if step1 > 0 and step2 > 0 else 0
    if log_steps_diff != 0:
        slope_per_decade = ((loss2 - loss1) / loss1) / log_steps_diff
        slope_percent = slope_percent = slope_per_decade * 100  # Convert to percentage
    else:
        slope_percent = float('inf')
    
    print(f"{step1} -> {step2}: {loss1:.6f} -> {loss2:.6f} | {slope_percent:.2f}%/decade")

    # Get all steps and losses up to the current point
    steps = df[step_column].values[:idx2+1].astype(float)
    losses = df[loss_column].values[:idx2+1]
    
    # Only use points where step > 0 to avoid log(0)
    mask = steps > 0
    if not np.any(mask):
        print("No valid steps > 0 for power law fitting")
        return
        
    steps = steps[mask]
    losses = losses[mask]
    
    # Define the power law function: L(t) = A * t^(-gamma) + C
    def power_law(t, A, gamma, C):
        return A * (t ** -gamma) + C
    
    try:
        # Initial parameter guesses
        p0 = [losses[0], 0.5, losses[-1]]
        
        # Fit the power law to all points up to idx2-1
        popt, _ = curve_fit(
            power_law,
            steps[:-1],
            losses[:-1],
            p0=p0,
            bounds=([0, 0, 0], [np.inf, 10, np.inf])
        )
        
        # Predict the next point
        predicted_loss = power_law(steps[-1], *popt)
        actual_loss = losses[-1]
        error = predicted_loss - actual_loss
        relative_error = error / actual_loss if actual_loss != 0 else float('inf')
        
        print(f"\nPower Law Fit: L(t) = {popt[0]:.2e} * t^(-{popt[1]:.3f}) + {popt[2]:.4f}")
        print(f"Predicted: {predicted_loss:.6f}, Actual: {actual_loss:.6f}")
        print(f"Error: {error:.2e} ({relative_error*100:.1f}%)")
        
        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Plot raw data points
        plt.scatter(steps, losses, color='lightgray', alpha=0.5, label='Raw Data')
        plt.plot(steps, losses, 'b-', linewidth=2, label='Data')
        
        # Plot power law fit
        t_plot = np.linspace(steps[0], steps[-1], 1000)
        power_law_fit = power_law(t_plot, *popt)
        plt.plot(t_plot, power_law_fit, 'r--', linewidth=2, 
                label=f'Power Law Fit: {popt[0]:.2e}·t^(-{popt[1]:.3f}) + {popt[2]:.4f}')
        
        # Highlight the predicted point
        plt.scatter([steps[-1]], [actual_loss], color='red', s=100, 
                   label=f'Actual: {actual_loss:.6f}\nPredicted: {predicted_loss:.6f}')
        
        plt.xscale('log')
        plt.xlabel('Step (log scale)')
        plt.ylabel('Loss')
        plt.title('Power Law Fit vs Smoothed Data')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"\nPower law fitting failed: {str(e)}")
        if 'popt' in locals():
            print(f"Last successful parameters: A={popt[0]:.2e}, gamma={popt[1]:.3f}, C={popt[2]:.4f}")

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
    
    # Show first few rows for reference
    print("\nFirst few data points:")
    print(df[[step_column, loss_column]].head())
    
    while True:
        try:
            print("\nEnter two indices to analyze (or 'q' to quit):")
            user_input = input("Indices (start end): ").strip()
            
            if user_input.lower() == 'q':
                break
                
            idx1, idx2 = map(int, user_input.split())
            analyze_two_points(df, loss_column, step_column, idx1, idx2)
            
        except ValueError:
            print("Please enter two integers separated by a space.")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    files = ["/Users/arnavshah/Code/dnaBLT/run_curves/wandb_export_2025-05-28T17_51_03.274-04_00.csv"]
    # files = ["/Users/arnavshah/Code/dnaBLT/run_curves/wandb_export_2025-05-28T18_09_04.251-04_00.csv"]
    
    for file_path in files:
        if os.path.exists(file_path):
            analyze_file(file_path)
        else:
            print(f"File not found: {file_path}")
            return
        
        # Ask if user wants to analyze another file
        if len(files) > 1:
            another = input("\nAnalyze another file? (y/n): ").strip().lower()
            if another != 'y':
                break

if __name__ == "__main__":
    main()
