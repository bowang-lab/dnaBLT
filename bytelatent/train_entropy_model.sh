#!/bin/bash
#SBATCH -c 5
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=entropy_model
#SBATCH --mem=32GB
#SBATCH --output=%x-%j.out
#SBATCH -p a40
#SBATCH --time=16:00:00

# Activate environment
conda activate blt

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run training
python train_entropy_model.py \
    --data_path /path/to/data \
    --hidden_dim 512 \
    --n_layers 14 \
    --n_heads 8 \
    --seq_length 2048 \
    --ffn_dim_multiplier 4 \
    --sliding_window 512 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --max_epochs 10 \
    --grad_clip 1.0 \
    --run_name "entropy_model_$(date +%Y%m%d_%H%M%S)"
