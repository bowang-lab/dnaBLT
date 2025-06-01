#!/bin/bash
#SBATCH --job-name=dnaBLT
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=18:00:00
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gres=gpu:2
#SBATCH -c 20

# conda activate blt
# home_dir="/home/ashah/byte-latent-stripedhyena2"
# cd $home_dir
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

stdbuf -oL -eL srun --exclusive torchrun --standalone --nproc_per_node=2 lightning_train.py --tokens 36000000000 --batch_size 32 --patch_size 2 --lr 0.0003 --dim_global 448 --dim_local 256 --global_layers 7 --decoder_layers 2 --num_gpus 2