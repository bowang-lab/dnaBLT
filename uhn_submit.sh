#!/bin/bash
#SBATCH -c 64
#SBATCH --gres=gpu:4
#SBATCH --job-name=dnaBLT
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

# Set directories
home_dir="/cluster/home/t136151uhn/dnaBLT"
data_path="/cluster/projects/bwanggroup/open-genome"
output_path="/cluster/projects/bwanggroup/dnaBLT"
cd $home_dir
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

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

stdbuf -oL -eL srun --exclusive python training/data/iterators/lightning_train.py --num_gpus 4 --tokens 55091200000 --batch_size 32 --grad_accum_size 4 --patch_size 2 --lr 0.0009 --dim_global 384 --dim_local 192 --global_layers 6 --decoder_layers 2
