#!/bin/bash
#SBATCH -c 100
#SBATCH --gres=gpu:4
#SBATCH --job-name=entropy_model
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -t 3-00:0:0
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=400G
#SBATCH --reservation=h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

# Source conda
source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
eval "$(conda shell.bash hook)"

# Activate environment
conda activate blt
echo "Using $PYTHON_PATH"

# Set directories
home_dir="/cluster/home/t136151uhn/dnaBLT"
data_path="/cluster/projects/bwanggroup/open-genome"
output_path="/cluster/projects/bwanggroup/dnaBLT"
cd $home_dir

# Set the master node address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=29500
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

# Print some information
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"1~# Set the master node address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# export MASTER_PORT=29500
export MASTER_PORT=$(python - <<EOF
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('', 0))  # OS will allocate a free port
free_port = sock.getsockname()[1]
sock.close()
print(free_port)
EOF
)

# Print some information
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


# Run training
stdbuf -oL -eL srun --exclusive python3 bytelatent/train_entropy_model.py \
    --data_path $data_path \
    --checkpoint_dir $output_path/checkpoints \
    --stage stage1 \
    --hidden_dim 512 \
    --n_layers 14 \
    --n_heads 8 \
    --seq_length 8192 \
    --ffn_dim_multiplier 4 \
    --sliding_window 512 \
    --batch_size 16 \
    --num_workers 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --max_epochs 2 \
    --grad_clip 1.0 \
    --grad_accum 2 \
    --devices 4 \
    --strategy deepspeed_stage_3 \
    --run_name "entropy_model_$(date +%Y%m%d_%H%M%S)" \
    --seed 23 \
    --join_stage_path True
