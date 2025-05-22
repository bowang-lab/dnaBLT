#!/bin/bash
#SBATCH --job-name=dnaBLT
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=14:00:00
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gpus=4

# conda activate blt
# home_dir="/home/ashah/byte-latent-stripedhyena2"
# cd $home_dir
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"

stdbuf -oL -eL srun --exclusive torchrun --standalone --nproc_per_node=4 lightning_train.py --tokens 19900000000 --batch_size 16 --grad_accum_size 8 --patch_size 2 --lr 0.0009 --dim_global 512 --dim_local 320 --global_layers 10 --decoder_layers 3 --num_gpus 4