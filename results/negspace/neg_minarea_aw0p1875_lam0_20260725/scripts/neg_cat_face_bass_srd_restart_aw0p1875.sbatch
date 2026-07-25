#!/bin/bash
#SBATCH --job-name=neg_cat_face_bass_srd_restart_aw0p1875
#SBATCH --partition=3090-gcondo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/negspace/neg_minarea_aw0p1875_lam0_20260725/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/negspace/neg_minarea_aw0p1875_lam0_20260725/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_final.py --target1 /oscar/home/cjmok/aps-hinge2/images/cat_face.png --target2 /oscar/home/cjmok/aps-hinge2/images/bass.png --arm srd --overlap-mode planar --seed 0 --steps 20000 --early-stop --min-steps 1500 --patience-steps 300 --eval-interval 25 --render-every 10 --render-every-scale 2 --n-patches 20 --swept-resolution 300 --swept-spawn-fraction 1.0 --srd-candidates 64 --srd-min-patch-area 0.005 --lambda-count 0 --silhouette-weight 0.710526 --negative-space-weight 3.78947 --render-scale 4 --max-hours 23.0 --device cuda --output-dir /oscar/home/cjmok/aps-hinge2/results/negspace/neg_minarea_aw0p1875_lam0_20260725/neg_cat_face_bass_srd_restart_aw0p1875 --area-normalized-view-loss --conflict-restart
