#!/bin/bash
#SBATCH --job-name=co_sun_moon
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/countobj/co_20260722_1223/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/countobj/co_20260722_1223/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/sun.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/moon.png --arm srd --count-objective --count-target-iou 0.9 --count-lambda 2000 --count-power 2 --count-softplus-beta 0 --overlap-mode planar --seed 0 --steps 4000 --early-stop --min-steps 1000 --patience-steps 300 --eval-interval 25 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 3 --negative-space-weight 1.5 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/countobj/co_20260722_1223/co_sun_moon
