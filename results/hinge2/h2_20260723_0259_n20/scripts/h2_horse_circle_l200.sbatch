#!/bin/bash
#SBATCH --job-name=h2_horse_circle_l200
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/hinge2/h2_20260723_0259_n20/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/hinge2/h2_20260723_0259_n20/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/horse.png --target2 /oscar/home/cjmok/aps-hinge2/images/circle.png --arm srd --count-objective --count-target-iou 0.91 --count-power 1 --count-margin-reward 5 --count-hard-floor --count-start-mode on_target --count-start-margin 0.02 --count-start-patience 2 --delete-eval-steps 15 --shed-patience-evals 8 --count-lambda 200 --overlap-mode planar --seed 0 --steps 3000 --early-stop --min-steps 1000 --patience-steps 300 --eval-interval 25 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 3 --negative-space-weight 1.5 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/hinge2/h2_20260723_0259_n20/h2_horse_circle_l200
