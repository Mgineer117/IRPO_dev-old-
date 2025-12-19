#!/bin/bash
#SBATCH --job-name=IRPO-Ant1
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --time=2-00:00:00
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

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name irpo --irpo-type irpo --project Ant-test --rendering &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name irpo --irpo-type irpo --project Ant-test --rendering &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v2 --algo-name irpo --irpo-type irpo --project Ant-test --rendering --num-inner-updates 10 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v3 --algo-name irpo --irpo-type irpo --project Ant-test --rendering --num-inner-updates 10 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name antmaze-v2 --algo-name irpo --irpo-type irpo --project Ant-test --rendering --num-inner-updates 15 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name antmaze-v3 --algo-name irpo --irpo-type irpo --project Ant-test --rendering --num-inner-updates 15 &


# === Wait for all background jobs to finish ===
wait