#!/bin/bash
#SBATCH --job-name=h2c_water_fire_prune
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/h2compare/h2c_20260723_1050/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/h2compare/h2c_20260723_1050/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/water.png --target2 /oscar/home/cjmok/aps-hinge2/images/fire.png --arm srd --lambda-count 0.0001 --overlap-mode planar --seed 0 --steps 3000 --early-stop --min-steps 1000 --patience-steps 300 --eval-interval 10 --render-every 10 --render-every-scale 1 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 0.9 --negative-space-weight 3.6 --area-normalized-view-loss --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/h2compare/h2c_20260723_1050/h2c_water_fire_prune --prune-path --prune-min-pieces 1 --prune-refit-steps 4 --prune-tolerance 0.01
