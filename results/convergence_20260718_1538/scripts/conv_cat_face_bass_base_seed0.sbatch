#!/bin/bash
#SBATCH --job-name=conv_cat_face_bass_base_seed0
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/convergence_20260718_1538/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/convergence_20260718_1538/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python compare_convergence.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/cat_face.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/bass.png --method base --seed 0 --n-patches 20 --swept-resolution 256 --max-steps 5000 --max-hours 10.0 --eval-interval 25 --plateau-delta 0.002 --plateau-window 250 --plateau-patience 2 --target-iou 0.85 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/convergence_20260718_1538/conv_cat_face_bass_base_seed0
