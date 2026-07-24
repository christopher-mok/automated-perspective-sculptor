#!/bin/bash
#SBATCH --job-name=h2c_horse_circle_hinge2
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/h2compare/h2c_20260723_2150/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/h2compare/h2c_20260723_2150/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/horse.png --target2 /oscar/home/cjmok/aps-hinge2/images/circle.png --arm srd --lambda-count 0 --overlap-mode planar --seed 0 --steps 3000 --early-stop --min-steps 1000 --patience-steps 300 --eval-interval 10 --render-every 10 --render-every-scale 1 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 0.9 --negative-space-weight 3.6 --area-normalized-view-loss --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/h2compare/h2c_20260723_2150/h2c_horse_circle_hinge2 --count-objective --count-target-iou 0.91 --count-power 1 --count-margin-reward 5 --count-hard-floor --count-start-mode on_target --count-start-margin 0.02 --count-start-patience 2 --delete-eval-steps 15 --shed-patience-evals 8 --count-lambda 500
