#!/bin/bash
#SBATCH --job-name=fanfill_sun_moon_basic_seed2
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/fanfill_20260720_0241/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/fanfill_20260720_0241/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python compare_fanfill.py --target1 /oscar/home/cjmok/automated-perspective-sculptor/images/sun.png --target2 /oscar/home/cjmok/automated-perspective-sculptor/images/moon.png --method basic --seed 2 --steps 2000 --eval-interval 25 --n-patches 20 --swept-resolution 256 --max-hours 10.0 --device cuda --output-dir /oscar/home/cjmok/automated-perspective-sculptor/results/fanfill_20260720_0241/fanfill_sun_moon_basic_seed2
