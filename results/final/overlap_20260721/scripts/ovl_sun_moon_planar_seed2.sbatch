#!/bin/bash
#SBATCH --job-name=ovl_sun_moon_planar_seed2
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/final/overlap_20260721/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/final/overlap_20260721/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/sun.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/moon.png --arm srd --overlap-mode planar --seed 2 --steps 2000 --eval-interval 25 --n-patches 20 --swept-resolution 256 --swept-spawn-fraction 1.0 --silhouette-weight 3.6 --negative-space-weight 0.9 --max-hours 10.0 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/final/overlap_20260721/ovl_sun_moon_planar_seed2
