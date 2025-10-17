#!/bin/bash
#SBATCH --job-name=IRPO
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:3
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fourrooms-v0 --project IRPO --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project IRPO --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v1 --project IRPO --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-v0 --project IRPO --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-v1 --project IRPO --algo-name ppo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fetchreach-v0 --project IRPO --algo-name ppo &

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fourrooms-v0 --project IRPO --algo-name drndppo &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --project IRPO --algo-name drndppo &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v1 --project IRPO --algo-name drndppo &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-v0 --project IRPO --algo-name drndppo &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-v1 --project IRPO --algo-name drndppo &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetchreach-v0 --project IRPO --algo-name drndppo &

CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name fourrooms-v0 --project IRPO --algo-name hrl &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project IRPO --algo-name hrl &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v1 --project IRPO --algo-name hrl &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name pointmaze-v0 --project IRPO --algo-name hrl &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name pointmaze-v1 --project IRPO --algo-name hrl &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name fetchreach-v0 --project IRPO --algo-name hrl &

# === Wait for all background jobs to finish ===
wait