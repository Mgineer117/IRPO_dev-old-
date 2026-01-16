#!/bin/bash
#SBATCH --job-name=Ant1
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=10
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

export MUJOCO_GL=osmesa

echo "Starting SEED 0..."


CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --algo-name irpo --irpo-type irpo --intrinsic-reward-mode random --project Ant-ablation --seed 1 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --algo-name irpo --irpo-type irpo --intrinsic-reward-mode random --project Ant-ablation --seed 2 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --algo-name irpo --irpo-type irpo --intrinsic-reward-mode random --project Ant-ablation --seed 3 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --algo-name irpo --irpo-type irpo --intrinsic-reward-mode random --project Ant-ablation --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --algo-name irpo --irpo-type irpo --intrinsic-reward-mode random --project Ant-ablation --seed 5 --num-runs 1 &
# # Wait for all Seed 0 jobs to finish before continuing
# wait 
# echo "Finished SEED 0. Starting SEED 1..."

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --algo-name hrl --intrinsic-reward-mode random --project Ant-ablation --seed 1 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --algo-name hrl --intrinsic-reward-mode random --project Ant-ablation --seed 2 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --algo-name hrl --intrinsic-reward-mode random --project Ant-ablation --seed 3 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --algo-name hrl --intrinsic-reward-mode random --project Ant-ablation --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --algo-name hrl --intrinsic-reward-mode random --project Ant-ablation --seed 5 --num-runs 1 &

# === Wait for all background jobs to finish ===
wait