"""Regenerate the SLURM job files in this folder.

Creates one .sh payload per experiment:

    ablation_<pair>_<setting>.sh      full / ablation        (submit_ablations.py settings)
    comparison_<pair>_<setting>.sh    basic / srd / annealing (submit_comparison.py settings)

The files contain only the job payload (cd to repo, activate env, run
run_ablation.py); cluster resources (partition/GPU/time/mem) are attached at
submission time by submit_job.py presets.

Rerun after changing image pairs or run settings:

    python slurm/generate_jobs.py [--trials 3] [--steps 2000]
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

_SLURM_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SLURM_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from submit_ablations import IMAGE_PAIRS, RUN_SETTINGS as ABLATION_SETTINGS  # noqa: E402
from submit_comparison import RUN_SETTINGS as COMPARISON_SETTINGS  # noqa: E402

JOB_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail
cd "${{SLURM_SUBMIT_DIR:-$(pwd)}}"
# module load cuda  # uncomment/adjust for your cluster
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

{command}
"""

EXPERIMENTS = {
    "ablation": (ABLATION_SETTINGS, ["full", "ablation"]),
    "comparison": (COMPARISON_SETTINGS, ["basic", "srd", "annealing"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    (_SLURM_DIR / "logs").mkdir(exist_ok=True)
    written = []
    for experiment, (settings, defaults) in EXPERIMENTS.items():
        for pair, (target1, target2) in IMAGE_PAIRS.items():
            for setting in defaults:
                job_name = f"{experiment}_{pair}_{setting}"
                command = shlex.join([
                    "python", "run_ablation.py",
                    "--target1", f"images/{target1}",
                    "--target2", f"images/{target2}",
                    *settings[setting],
                    "--trials", str(args.trials),
                    "--steps", str(args.steps),
                    "--device", "cuda",
                    "--output-dir", f"results/slurm/{job_name}",
                ])
                path = _SLURM_DIR / f"{job_name}.sh"
                path.write_text(
                    JOB_TEMPLATE.format(job_name=job_name, command=command),
                    encoding="utf-8",
                )
                written.append(path.name)

    print(f"Wrote {len(written)} job files to {_SLURM_DIR}/:")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
