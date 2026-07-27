#!/bin/bash
#SBATCH --job-name=final_horse_circle_basic_seed0
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/final/baseline_initpc_20260726/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/final/baseline_initpc_20260726/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/horse.png --target2 /oscar/home/cjmok/aps-hinge2/images/circle.png --arm basic --overlap-mode planar --seed 0 --steps 20000 --eval-interval 25 --n-patches 13 --swept-resolution 300 --swept-spawn-fraction 1.0 --silhouette-weight 2 --negative-space-weight 2.5 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/final/baseline_initpc_20260726/final_horse_circle_basic_seed0 --render-every 10 --render-every-scale 2 --early-stop --min-steps 1500 --patience-steps 300
