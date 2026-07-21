#!/bin/bash
#SBATCH --job-name=conv_horse_circle_srd_seed0
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/convergence_canary3/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/convergence_canary3/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python compare_convergence.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/horse.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/circle.png --method srd --seed 0 --n-patches 20 --swept-resolution 64 --max-steps 400 --max-hours 0.5 --eval-interval 25 --plateau-delta 0.002 --plateau-window 250 --plateau-patience 2 --target-iou 0.8 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/convergence_canary3/conv_horse_circle_srd_seed0
