#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=6
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

export MUJOCO_GL=osmesa

# === Run Experiments in Parallel ===
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name hrl --project Ant --seed 3 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v0 --algo-name hrl --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --algo-name hrl --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name hrl --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name hrl --project Ant --seed 4 --num-runs 1 &

wait
echo "All seeds finished."