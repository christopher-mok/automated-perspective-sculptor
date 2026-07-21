#!/bin/bash
#SBATCH --job-name=hard_teapot_droplets_srd_no_swept_seed0
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/hard_experiments/hard_20260721_1354/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/hard_experiments/hard_20260721_1354/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/teapot.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/droplets.png --arm srd_no_swept --overlap-mode planar --seed 0 --steps 4000 --early-stop --min-steps 1000 --patience-steps 300 --eval-interval 25 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 3 --negative-space-weight 1.5 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/hard_experiments/hard_20260721_1354/hard_teapot_droplets_srd_no_swept_seed0
