import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from collections import deque
import plotly.graph_objects as go
import plotly.io as pio

from training.data.iterators.lightning_train import (
    PackedBatchDataset,
    compute_loss,
    to_device_async,
)

from bytelatent.model.blt import ByteLatentTransformer

from training.data.iterators.v_args import DataloaderArgs, TrainArgs

def run_experiment(checkpoint_path, shuffle, device):
    # Load training arguments
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint["hyper_parameters"] if "hyper_parameters" in checkpoint else checkpoint["args"]
    if isinstance(train_args, dict):
        train_args = TrainArgs(**train_args)

    # Build datamodule with custom shuffle
    train_args.data.sources = {"train": {"16b[34].arrow": 1}, "validation": {"entropies_validation.arrow": 1}}
    train_dataset = PackedBatchDataset(train_args.data, dataset_key="train", shuffle=shuffle)
    train_loader = DataLoader(train_dataset, batch_size=None)

    # Build model and load weights
    model = ByteLatentTransformer(train_args.model)
    model.to(device)
    model.eval()
    model.load_state_dict(checkpoint["state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    running_gnorms = []

    step = 0
    for batch_cpu in train_loader:
        if step >= 50:
            break

        batch = to_device_async(batch_cpu, device)
        optimizer.zero_grad()

        # Forward
        pred = model(batch.x, batch.patch_lengths, batch.ngram_ids)
        loss, tok_loss = compute_loss(pred, batch.y, batch.mask, scale=1.0)

        # Backward
        loss.backward()

        g_norm = torch.sqrt(sum(p.grad.data.norm()**2
                        for p in model.parameters()
                        if p.grad is not None))
        running_gnorms.append(g_norm.item())

        # Optimizer step
        optimizer.step()

        step += 1

    mean_g  = np.mean(running_gnorms)
    var_g   = np.var(running_gnorms, ddof=1)           # unbiased
    noise_S = var_g / (mean_g**2)

    return running_gnorms, mean_g, var_g, noise_S

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running experiment with shuffle=False...")
    gnorms_no_shuffle, mean_no_shuffle, var_no_shuffle, noise_S_no_shuffle = run_experiment(args.checkpoint, False, device)
    print(f"Shuffle=False | Mean ∥g∥₂: {mean_no_shuffle:.3f} | Var: {var_no_shuffle:.3f} | Noise scale: {noise_S_no_shuffle:.3f}")

    print("\nRunning experiment with shuffle=True...")
    gnorms_shuffle, mean_shuffle, var_shuffle, noise_S_shuffle = run_experiment(args.checkpoint, True, device)
    print(f"Shuffle=True | Mean ∥g∥₂: {mean_shuffle:.3f} | Var: {var_shuffle:.3f} | Noise scale: {noise_S_shuffle:.3f}")

    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=gnorms_no_shuffle, mode='lines', name='No Shuffle'))
    fig.add_trace(go.Scatter(y=gnorms_shuffle, mode='lines', name='Shuffle'))

    fig.update_layout(
        title="Running Gradient Norms (L2)",
        xaxis_title="Step",
        yaxis_title="Gradient Norm (L2)",
        hovermode="x unified"
    )

    plot_filename = "running_gnorms_plot.html"
    pio.write_html(fig, plot_filename)
    print(f"\nPlot saved to {plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch Lightning checkpoint (.ckpt)")
    args = parser.parse_args()
    main(args)