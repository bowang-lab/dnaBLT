#!/bin/bash
#SBATCH -c 100
#SBATCH --gres=gpu:4
#SBATCH --job-name=entropy_model
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -t 1-00:0:0
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=400G
#SBATCH --reservation=h100

# Source conda
source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
eval "$(conda shell.bash hook)"

# Activate environment
conda activate blt
echo "Using $PYTHON_PATH"

# Set directories (THIS IS ashah01'S T-ID MAKE SURE TO USE YOUR OWN)
home_dir="/cluster/home/t136151uhn/dnaBLT"
checkpoints_dir="/cluster/projects/bwanggroup/dnaBLT/checkpoints"
data_path="/cluster/projects/bwanggroup/open-genome"
cd $home_dir

# Set the master node address (first node in the allocation)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
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
stdbuf -oL -eL srun --exclusive python3 compute_entropies/auxiliary_entropy.py \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --backend nccl \
    --world_size 4 \
    --gpu_per_node 4 \
    --data_path $data_path \
    --data_cache_dir $data_path/cache \
    --split test \
    --batch_size 4 \
    --arrow_batch 10 \
    --entropy_model_checkpoint_dir $checkpoints_dir/last.ckpt