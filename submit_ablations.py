"""Send ablation experiments to a SLURM cluster as run_ablation.py jobs.

Expands a sweep (image pairs x run settings), writes one sbatch script per
job, submits them, and records a manifest. Each job is a plain
``run_ablation.py`` invocation, so its results land in
``<sweep-dir>/<job-name>/output.txt`` exactly like a local run.

Run settings
------------
    full             complete method
    ablation         all three components disabled
    no-swept-adds    only swept-volume-guided additions disabled
    loss-only-del    only rule-based deletion disabled
    no-splitting     only splitting disabled

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_ablations.py --sweep-name srd_ablation --dry-run

    # Submit full vs. ablation on every benchmark pair, 3 trials each
    python submit_ablations.py --sweep-name srd_ablation --trials 3 --steps 500

    # Single-component ablations on one pair, custom cluster resources
    python submit_ablations.py --sweep-name components --pairs horse_circle \
        --settings no-swept-adds loss-only-del no-splitting \
        --partition gpu --gres gpu:1 --time 04:00:00

    # No scheduler available: run the same sweep sequentially here
    python submit_ablations.py --sweep-name srd_ablation --local

    # After jobs finish: aggregate every output.txt into one table
    python submit_ablations.py --sweep-name srd_ablation --collect
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_PAIRS: dict[str, tuple[str, str]] = {
    "sun_moon": ("sun.png", "moon.png"),
    "horse_circle": ("horse.png", "circle.png"),
    "cat_face_bass": ("cat_face.png", "bass.png"),
    "water_fire": ("water.png", "fire.png"),
    "acm_scf": ("acm.png", "scf.png"),
    # Harder pairs: two detailed silhouettes rather than one shape against a
    # simple primitive, so both views compete for the same geometry.
    "siggraph_sigchi": ("siggraph.png", "sigchi.png"),
    "crane_crab": ("crane.png", "crab.png"),
    "teapot_droplets": ("teapot.png", "droplets.png"),
    "dancer_guitar": ("dancer.png", "guitar.png"),
    "dance_argument": ("dance.png", "argument.png"),
    "robot_man": ("robot.png", "man.png"),
    # Second batch of targets. axe and dance each appear in two pairs, so the
    # same silhouette is asked to share geometry with two different partners
    # and the pairs are not independent of each other.
    "tshirt_pants": ("tshirt.png", "pants.png"),
    "axe_shield": ("axe.png", "shield.png"),
    "dance_sitting": ("dance.png", "sitting.png"),
    "happy_sad": ("happy.png", "sad.png"),
    "axe_tree": ("axe.png", "tree.png"),
    "guitar_dance": ("guitar.png", "dance.png"),
}

# Run setting -> extra run_ablation.py flags.
RUN_SETTINGS: dict[str, list[str]] = {
    "full": ["--mode", "full"],
    "ablation": ["--mode", "ablation"],
    "no-swept-adds": ["--mode", "full", "--no-swept-volume-adds"],
    "loss-only-del": ["--mode", "full", "--loss-only-deletion"],
    "no-splitting": ["--mode", "full", "--no-splitting"],
}

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{gres_line}#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err

set -euo pipefail
cd {project_root}
# Relax errexit/nounset while sourcing shell-init and conda: Oscar's
# /etc/bashrc references unbound vars (e.g. BASHRCSOURCED) and conda
# activate can return non-zero, either of which aborts the job otherwise.
set +eu
{env_setup}
set -eu
{command}
"""


def _parse_args(
    run_settings: dict[str, list[str]],
    description: str,
    default_settings: list[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-name", required=True, help="Name for this batch of jobs.")
    parser.add_argument(
        "--pairs",
        nargs="*",
        choices=sorted(IMAGE_PAIRS),
        default=sorted(IMAGE_PAIRS),
        help="Image pairs to run.",
    )
    parser.add_argument(
        "--settings",
        nargs="*",
        choices=sorted(run_settings),
        default=default_settings,
        help="Run settings to compare.",
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per job.")
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps per trial.")
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3.5e-3)
    parser.add_argument("--device", default="cuda", help="Device passed to run_ablation.py.")
    parser.add_argument(
        "--split-trials",
        action="store_true",
        help="One job per seed instead of one multi-trial job (more parallelism).",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra flags forwarded verbatim to run_ablation.py, quoted as one "
             "string (e.g. \"--swept-resolution 64 --no-renders\").",
    )

    cluster = parser.add_argument_group("cluster resources")
    cluster.add_argument("--partition", default="gpu")
    cluster.add_argument("--gres", default="gpu:1", help="Empty string for CPU-only jobs.")
    cluster.add_argument("--time", default="06:00:00", help="SLURM walltime limit.")
    cluster.add_argument("--mem", default="16G")
    cluster.add_argument("--cpus", type=int, default=4)
    cluster.add_argument(
        "--python",
        default="python",
        help="Python executable on the cluster (e.g. .venv/bin/python).",
    )
    cluster.add_argument(
        "--env-setup",
        default="",
        help="Shell line(s) run before the job, e.g. "
             "\"module load cuda/12.8 && source .venv/bin/activate\".",
    )

    mode = parser.add_argument_group("actions")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sbatch scripts and the manifest but do not submit.",
    )
    mode.add_argument(
        "--local",
        action="store_true",
        help="Run every job sequentially on this machine instead of submitting.",
    )
    mode.add_argument(
        "--collect",
        action="store_true",
        help="Aggregate output.txt results from a finished sweep and exit.",
    )
    return parser.parse_args()


def _build_jobs(
    args: argparse.Namespace,
    sweep_dir: Path,
    run_settings: dict[str, list[str]],
) -> list[dict]:
    """Expand the sweep into one dict per job with its command line."""
    image_dir = _PROJECT_ROOT / "images"
    seed_groups: list[list[int] | None]
    if args.split_trials:
        seed_groups = [[seed] for seed in range(args.trials)]
    else:
        seed_groups = [None]  # one job carrying all trials

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for setting in args.settings:
            for seeds in seed_groups:
                suffix = f"_seed{seeds[0]}" if seeds is not None else ""
                job_name = f"{pair}_{setting}{suffix}"
                output_dir = sweep_dir / job_name
                command = [
                    args.python, "run_ablation.py",
                    "--target1", str(image_dir / target1),
                    "--target2", str(image_dir / target2),
                    *run_settings[setting],
                    "--steps", str(args.steps),
                    "--n-patches", str(args.n_patches),
                    "--lr", str(args.lr),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                if seeds is not None:
                    command += ["--seeds", *map(str, seeds)]
                else:
                    command += ["--trials", str(args.trials)]
                if args.extra_args:
                    command += shlex.split(args.extra_args)
                jobs.append({
                    "name": job_name,
                    "pair": pair,
                    "setting": setting,
                    "output_dir": output_dir,
                    "command": command,
                })
    return jobs


def _write_sbatch_script(job: dict, args: argparse.Namespace, sweep_dir: Path) -> Path:
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gres_line = f"#SBATCH --gres={args.gres}\n" if args.gres else ""
    script = SBATCH_TEMPLATE.format(
        job_name=job["name"],
        partition=args.partition,
        gres_line=gres_line,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        log_dir=log_dir,
        project_root=_PROJECT_ROOT,
        env_setup=args.env_setup,
        command=shlex.join(job["command"]),
    )
    script_path = sweep_dir / "scripts" / f"{job['name']}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _submit(script_path: Path) -> str:
    """sbatch the script and return the SLURM job id."""
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _run_local(job: dict) -> int:
    print(f"[Local] {job['name']}: {shlex.join(job['command'])}")
    return subprocess.run(job["command"], cwd=_PROJECT_ROOT, check=False).returncode


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\tsetting\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{row['setting']}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def _collect(sweep_dir: Path) -> None:
    """Aggregate mean/std final loss from every job's output.txt."""
    reports = sorted(sweep_dir.glob("*/output.txt"))
    if not reports:
        print(f"No output.txt files found under {sweep_dir}")
        return

    print(f"{'job':<40} {'trials':>6} {'mean_loss':>12} {'std':>10} {'mean_s':>10}")
    for report in reports:
        text = report.read_text(encoding="utf-8")

        def _grab(key: str) -> str:
            match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
            return match.group(1).strip() if match else "-"

        print(
            f"{report.parent.name:<40} "
            f"{_grab('completed_trials'):>6} "
            f"{_grab('mean_final_loss'):>12} "
            f"{_grab('std_final_loss'):>10} "
            f"{_grab('mean_trial_seconds'):>10}"
        )


def run_cli(
    run_settings: dict[str, list[str]],
    description: str,
    default_settings: list[str],
) -> None:
    """Shared submission CLI, parameterized by the available run settings."""
    args = _parse_args(run_settings, description, default_settings)
    sweep_dir = _PROJECT_ROOT / "results" / args.sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir, run_settings)
    print(
        f"[Sweep] {args.sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.settings)} setting(s)"
        f"{f' x {args.trials} seed(s)' if args.split_trials else ''})"
    )

    if args.local:
        failed = [job["name"] for job in jobs if _run_local(job) != 0]
        _write_manifest(sweep_dir, jobs)
        if failed:
            print(f"[Local] {len(failed)} job(s) failed: {', '.join(failed)}")
            sys.exit(1)
        print(f"[Local] all {len(jobs)} job(s) finished; try --collect for a summary")
        return

    submitted: list[dict] = []
    for job in jobs:
        script_path = _write_sbatch_script(job, args, sweep_dir)
        if args.dry_run:
            print(f"[Dry run] wrote {script_path}")
            submitted.append(job)
            continue
        job["job_id"] = _submit(script_path)
        print(f"[Submitted] {job['job_id']}: {job['name']}")
        submitted.append(job)

    manifest_path = _write_manifest(sweep_dir, submitted)
    print(f"[Sweep] manifest written to {manifest_path}")
    if not args.dry_run:
        print("Monitor with: squeue --me | grep " + args.sweep_name[:8])
        print(f"Aggregate when done: python {Path(sys.argv[0]).name} "
              f"--sweep-name {args.sweep_name} --collect")


def main() -> None:
    run_cli(
        RUN_SETTINGS,
        "Submit run_ablation.py ablation experiments to a SLURM cluster.",
        default_settings=["full", "ablation"],
    )


if __name__ == "__main__":
    main()
