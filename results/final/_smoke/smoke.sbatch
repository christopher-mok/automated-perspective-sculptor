#!/bin/bash
#SBATCH --job-name=final_smoke
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/automated-perspective-sculptor
#SBATCH --output=/oscar/home/cjmok/automated-perspective-sculptor/results/final/_smoke/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/automated-perspective-sculptor/results/final/_smoke/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu
ulimit -n 50000

echo "--- planar + repair, srd ---"
python run_final.py --target1 images/horse.png --target2 images/circle.png \
  --arm srd --overlap-mode planar --seed 0 --steps 60 --eval-interval 10 \
  --swept-resolution 128 --silhouette-weight 3.6 --negative-space-weight 0.9 \
  --output-dir results/final/_smoke/planar_srd

echo "--- sphere, no repair, srd_no_swept ---"
python run_final.py --target1 images/horse.png --target2 images/circle.png \
  --arm srd_no_swept --overlap-mode sphere --no-overlap-repair --seed 0 --steps 60 \
  --eval-interval 10 --swept-resolution 128 --silhouette-weight 3.6 \
  --negative-space-weight 0.9 --output-dir results/final/_smoke/sphere_nosw

echo "--- basic arm ---"
python run_final.py --target1 images/horse.png --target2 images/circle.png \
  --arm basic --overlap-mode planar --seed 0 --steps 60 --eval-interval 10 \
  --swept-resolution 128 --silhouette-weight 3.6 --negative-space-weight 0.9 \
  --output-dir results/final/_smoke/planar_basic

echo "ALL SMOKE RUNS DONE"
