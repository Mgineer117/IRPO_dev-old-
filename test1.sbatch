#!/bin/bash
#SBATCH --job-name=IRPO
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --project Ant --algo-name irpo --irpo-type irpo_pick --inner-actor-lr 1.0 --num-inner-updates 10 --target-kl 0.0005 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --project Ant --algo-name irpo --irpo-type irpo_pick --inner-actor-lr 0.05 --num-inner-updates 20 --target-kl 0.0005 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --project Ant --algo-name irpo --irpo-type irpo_pick --num-inner-updates 20 --target-kl 0.0005 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --project Ant --algo-name irpo --irpo-type irpo_pick --inner-actor-lr 1.0 --num-inner-updates 20 --target-kl 0.0005 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --project Ant --algo-name ppo &


# === Wait for all background jobs to finish ===
wait