#!/bin/bash
#SBATCH --job-name=dnaBLT
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=14:00:00
#SBATCH --partition=goodarzilab_gpu_priority
#SBATCH --gpus=2

conda activate blt
home_dir="/home/ashah/byte-latent-stripedhyena2"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

stdbuf -oL -eL srun --exclusive python training/data/iterators/lightning_train.py --tokens 17120000000 --batch_size 16 --grad_accum_size 8 --patch_size 2 --lr 0.0009 --dim_global 576 --dim_local 320 --global_layers 10 --decoder_layers 3 --global_heads 9 --local_heads 5