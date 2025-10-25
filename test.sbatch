#!/bin/bash
#SBATCH --job-name=IRPO-test
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=22
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v0 --algo-name irpo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --algo-name irpo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name irpo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name irpo --irpo-type irpo --project Ant &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v0 --algo-name ppo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --algo-name ppo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name ppo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name ppo --irpo-type irpo --project Ant &

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v0 --algo-name trpo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --algo-name trpo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name trpo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name trpo --irpo-type irpo --project Ant &

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v0 --algo-name drndppo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v1 --algo-name drndppo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v2 --algo-name drndppo --irpo-type irpo --project Ant &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v3 --algo-name drndppo --irpo-type irpo --project Ant &


CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-v0 --algo-name irpo --irpo-type irpo --project IRPO_random --intrinsic-reward-mode random &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-v1 --algo-name irpo --irpo-type irpo --project IRPO_random --intrinsic-reward-mode random &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetchreach-v0 --algo-name irpo --irpo-type irpo --project IRPO_random --intrinsic-reward-mode random &

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-v0 --algo-name hrl --project IRPO_random --intrinsic-reward-mode random &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-v1 --algo-name hrl --project IRPO_random --intrinsic-reward-mode random &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetchreach-v0 --algo-name hrl --project IRPO_random --intrinsic-reward-mode random &


# === Wait for all background jobs to finish ===
wait