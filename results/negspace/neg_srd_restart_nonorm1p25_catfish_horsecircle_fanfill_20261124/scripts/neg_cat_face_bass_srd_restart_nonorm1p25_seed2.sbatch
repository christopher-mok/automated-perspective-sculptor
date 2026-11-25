#!/bin/bash
#SBATCH --job-name=neg_cat_face_bass_srd_restart_nonorm1p25_seed2
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/negspace/neg_srd_restart_nonorm1p25_catfish_horsecircle_fanfill_20261124/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/negspace/neg_srd_restart_nonorm1p25_catfish_horsecircle_fanfill_20261124/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/cat_face.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/bass.png --arm srd --overlap-mode planar --seed 2 --steps 20000 --early-stop --min-steps 1500 --patience-steps 300 --eval-interval 25 --render-every 10 --render-every-scale 4 --n-patches 20 --swept-resolution 300 --swept-spawn-fraction 1.0 --srd-candidates 64 --srd-min-patch-area 0.01 --lambda-count 0.05 --lr 0.0035 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/negspace/neg_srd_restart_nonorm1p25_catfish_horsecircle_fanfill_20261124/neg_cat_face_bass_srd_restart_nonorm1p25_seed2 --render-oblique --silhouette-weight 2 --negative-space-weight 2.5 --conflict-restart
