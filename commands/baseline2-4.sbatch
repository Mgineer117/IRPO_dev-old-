#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=5
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

# === Run Experiments in Parallel ===
#!/bin/bash
echo "Starting SEED 0..."

# --- SEED 2 BLOCK ---
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v0 --algo-name drndppo --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v1 --algo-name drndppo --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name drndppo --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v3 --algo-name drndppo --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name antmaze-v2 --algo-name psne --project Ant --seed 2 --num-runs 1 &

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v0 --algo-name psne --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v1 --algo-name psne --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v2 --algo-name psne --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v3 --algo-name psne --project Ant --seed 4 --num-runs 1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name antmaze-v3 --algo-name psne --project Ant --seed 2 --num-runs 1 &


wait