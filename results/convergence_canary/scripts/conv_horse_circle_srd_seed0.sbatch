#!/bin/bash
#SBATCH --job-name=conv_horse_circle_srd_seed0
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/convergence_canary/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/convergence_canary/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python compare_convergence.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/horse.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/circle.png --method srd --seed 0 --n-patches 20 --swept-resolution 64 --max-steps 60 --max-hours 0.2 --eval-interval 10 --plateau-delta 0.005 --plateau-window 20 --plateau-patience 2 --target-iou 0.995 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/convergence_canary/conv_horse_circle_srd_seed0
