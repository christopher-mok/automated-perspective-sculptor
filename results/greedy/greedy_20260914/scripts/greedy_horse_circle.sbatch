#!/bin/bash
#SBATCH --job-name=greedy_horse_circle
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/oscar/home/cjmok/aps-hinge2
#SBATCH --output=/oscar/home/cjmok/aps-hinge2/results/greedy/greedy_20260914/logs/%x.%j.out
#SBATCH --error=/oscar/home/cjmok/aps-hinge2/results/greedy/greedy_20260914/logs/%x.%j.err

set +eu
source /oscar/home/cjmok/.bashrc
conda activate myenv
set -eu

ulimit -n 50000

python run_greedy.py --target1 /oscar/home/cjmok/aps-hinge2/images/horse.png --target2 /oscar/home/cjmok/aps-hinge2/images/circle.png --seed 0 --hanging-plane-size 5 --swept-resolution 256 --panel-half-size 0.12 --max-candidates 6000 --view1-weight 0.5 --view2-weight 0.5 --spill-weight 1 --min-gain 1 --max-panels 400 --output-dir /oscar/home/cjmok/aps-hinge2/results/greedy/greedy_20260914/greedy_horse_circle
