#!/bin/bash
#SBATCH --job-name=orig_cat_face_bass
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-original
#SBATCH --output=/oscar/home/cjmok/aps-original/results/original/orig_20260724/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-original/results/original/orig_20260724/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-original/images/cat_face.png --target2 /oscar/home/cjmok/aps-original/images/bass.png --arm srd --overlap-mode planar --overlap-repair --seed 0 --steps 20000 --early-stop --min-steps 1500 --patience-steps 300 --eval-interval 25 --render-every 10 --render-every-scale 2 --n-patches 20 --swept-resolution 300 --swept-spawn-fraction 1.0 --srd-candidates 64 --delete-eval-steps 1 --lambda-count 0 --silhouette-weight 2 --negative-space-weight 2.5 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-original/results/original/orig_20260724/orig_cat_face_bass --conflict-restart --conflict-eps 0.001
