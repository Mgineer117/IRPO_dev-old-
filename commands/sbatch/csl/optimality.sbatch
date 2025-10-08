#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fourrooms-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo --rendering & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fourrooms-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_is --rendering & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fourrooms-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_lr --rendering & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fourrooms-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_blend --rendering & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name fourrooms-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_pick --rendering & 

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo --rendering & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_is --rendering & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_lr --rendering & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_blend --rendering & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --project irpo_optimality --algo-name irpo --irpo-type irpo_pick --rendering &

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v1 --project irpo_optimality --algo-name irpo --irpo-type irpo --rendering & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v1 --project irpo_optimality --algo-name irpo --irpo-type irpo_is --rendering & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v1 --project irpo_optimality --algo-name irpo --irpo-type irpo_lr --rendering & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v1 --project irpo_optimality --algo-name irpo --irpo-type irpo_blend --rendering & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v1 --project irpo_optimality --algo-name irpo --irpo-type irpo_pick --rendering & 


# === Wait for all background jobs to finish ===
wait