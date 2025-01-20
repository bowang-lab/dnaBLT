#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos deadline
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH --job-name=entropy_model
#SBATCH --mem=64GB
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -p a40
#SBATCH --time=23:00:00
#SBATCH --no-requeue

# Source conda
source ~/.bashrc
eval "$(conda shell.bash hook)"

# Activate environment
conda activate blt
echo "Using $PYTHON_PATH"

# Set working directory
cd /h/afallah/dnaBLT

# Set environment variables
module load cuda-12.4

export CUDA_HOME=/pkgs/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Run training
stdbuf -oL -eL srun python3 bytelatent/train_entropy_model.py \
    --data_path /projects/llm/open-genome/ \
    --data_cache_dir /projects/llm/open-genome/cache \
    --stage stage1 \
    --hidden_dim 512 \
    --n_layers 14 \
    --n_heads 8 \
    --seq_length 8192 \
    --ffn_dim_multiplier 4 \
    --sliding_window 512 \
    --batch_size 6 \
    --num_workers 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --max_epochs 2 \
    --grad_clip 1.0 \
    --grad_accum 4 \
    --devices 4 \
    --strategy deepspeed_stage_3 \
    --run_name "entropy_model_$(date +%Y%m%d_%H%M%S)" \
    --seed 23

