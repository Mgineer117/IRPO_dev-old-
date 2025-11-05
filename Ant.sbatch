#!/bin/bash
#SBATCH --job-name=IRPO-test
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=IRPO.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate irpo

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name irpo --irpo-type irpo --project Ant --num-inner-updates 10 --target-kl 0.01 --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name irpo --irpo-type irpo --project Ant --num-inner-updates 10 --target-kl 0.01 --rendering &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name hrl --project Ant --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name hrl --project Ant --rendering &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name ppo --project Ant --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name ppo --project Ant --rendering &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name drndppo --project Ant --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name drndppo --project Ant --rendering &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name trpo --project Ant --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name trpo --project Ant --rendering &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name psne --project Ant --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name psne --project Ant --rendering &

# === Wait for all background jobs to finish ===
wait