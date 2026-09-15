"""Submit the greedy swept-volume square-packing baseline to SLURM.

A reviewer asked for a baseline that skips SRD and gradient descent
entirely: build the candidate set of fixed-size, fixed-yaw square panels
directly from the swept volume (so every candidate is two-view-valid by
construction), then greedily place panels by weighted marginal pixel
coverage gain minus a spill penalty (``optimizer/greedy_pack.py``,
``run_greedy.py``). This driver submits that method on the same five
standard pairs the negspace/hinge2 batches use
(cat_face_bass, horse_circle, robot_man, sun_moon, water_fire), one job per
pair, no seeds needed -- the method is deterministic given its config (the
only randomness is candidate subsampling when the swept volume yields more
points than --max-candidates, controlled by --seed).

The method needs no GPU and no torch autograd, so jobs run on a plain CPU
partition rather than the GPU partitions the rest of the codebase submits
to. This driver, run_greedy.py and optimizer/greedy_pack.py are the only
files involved -- nothing here is imported by run_final.py, core/optimizer.py
or optimizer/srd.py, so submitting this sweep cannot affect any other arm.

Examples
--------
    python submit_greedy.py --dry-run
    python submit_greedy.py
    python submit_greedy.py --sweep-name <name> --collect
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from submit_ablations import IMAGE_PAIRS

_PROJECT_ROOT = Path(__file__).resolve().parent
GREEDY_ROOT = _PROJECT_ROOT / "results" / "greedy"

# The standard five pairs used across the negspace/hinge2 batches (see
# [[h2ri-aw-and-grammar-ablation-batch]] and friends): one shape against a
# simple primitive in each view.
STANDARD_PAIRS = (
    "cat_face_bass", "horse_circle", "robot_man", "sun_moon", "water_fire",
)

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --chdir={project_root}
#SBATCH --output={log_dir}/%x.%j.out
#SBATCH --error={log_dir}/%x.%j.err

set +eu
{env_setup}
set -eu

ulimit -n 50000

{command}
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the greedy swept-volume square-packing baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-name", default=None,
                        help="Batch dir under results/greedy/ (default: greedy_<timestamp>).")
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Override the pair list (default: the standard 5 pairs).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Candidate-subsampling seed; the method is otherwise deterministic.")

    geometry = parser.add_argument_group("candidate geometry")
    geometry.add_argument("--hanging-plane-size", type=float, default=5.0)
    geometry.add_argument("--swept-resolution", type=int, default=256)
    geometry.add_argument("--panel-half-size", type=float, default=0.12)
    geometry.add_argument("--theta", type=float, default=None,
                          help="Fixed yaw override; default is the two-camera bisector.")
    geometry.add_argument("--max-candidates", type=int, default=6000)

    score = parser.add_argument_group("greedy scoring")
    score.add_argument("--view1-weight", type=float, default=0.5)
    score.add_argument("--view2-weight", type=float, default=0.5)
    score.add_argument("--spill-weight", type=float, default=1.0)
    score.add_argument("--min-gain", type=float, default=1.0)
    score.add_argument("--max-panels", type=int, default=400)

    cluster = parser.add_argument_group("cluster resources")
    cluster.add_argument("--partition", default="batch",
                         help="CPU partition -- this method needs no GPU.")
    cluster.add_argument("--time", default="01:00:00")
    cluster.add_argument("--mem", default="24G")
    cluster.add_argument("--cpus", type=int, default=4)
    cluster.add_argument("--python", default="python")
    cluster.add_argument("--env-setup",
                         default="source /oscar/home/cjmok/.bashrc\nconda activate myenv")

    mode = parser.add_argument_group("actions")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--collect", action="store_true")
    return parser.parse_args()


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        job_name = f"greedy_{pair}"
        output_dir = sweep_dir / job_name
        command = [
            args.python, "run_greedy.py",
            "--target1", str(image_dir / target1),
            "--target2", str(image_dir / target2),
            "--seed", str(args.seed),
            "--hanging-plane-size", f"{args.hanging_plane_size:g}",
            "--swept-resolution", str(args.swept_resolution),
            "--panel-half-size", f"{args.panel_half_size:g}",
            "--max-candidates", str(args.max_candidates),
            "--view1-weight", f"{args.view1_weight:g}",
            "--view2-weight", f"{args.view2_weight:g}",
            "--spill-weight", f"{args.spill_weight:g}",
            "--min-gain", f"{args.min_gain:g}",
            "--max-panels", str(args.max_panels),
            "--output-dir", str(output_dir),
        ]
        if args.theta is not None:
            command += ["--theta", f"{args.theta:g}"]
        jobs.append({
            "name": job_name, "pair": pair,
            "output_dir": output_dir, "command": command,
        })
    return jobs


def _write_sbatch_script(job: dict, args: argparse.Namespace, sweep_dir: Path) -> Path:
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = SBATCH_TEMPLATE.format(
        job_name=job["name"], partition=args.partition,
        time=args.time, mem=args.mem, cpus=args.cpus, log_dir=log_dir,
        project_root=_PROJECT_ROOT, env_setup=args.env_setup,
        command=shlex.join(job["command"]),
    )
    script_path = sweep_dir / "scripts" / f"{job['name']}.sbatch"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    return script_path


def _submit(script_path: Path) -> str:
    result = subprocess.run(["sbatch", str(script_path)],
                            capture_output=True, text=True, check=True)
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\toutput_dir"]
    for row in rows:
        lines.append(f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['output_dir']}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "n_candidates", "final_patches",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "final_mean_coverage", "final_mean_precision", "final_mean_spill",
    "total_seconds",
)


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1) if match else "-"


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[str]) -> str:
    floats = [v for v in (_safe_float(x) for x in values) if v is not None]
    return f"{sum(floats) / len(floats):.6g}" if floats else "-"


def _collect(sweep_dir: Path) -> None:
    reports = sorted(sweep_dir.glob("*/report.txt"))
    if not reports:
        print(f"No report.txt files found under {sweep_dir}")
        return

    collected = sweep_dir / "collected"
    collected.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for report in reports:
        text = report.read_text(encoding="utf-8")
        job_name = report.parent.name
        pair = job_name.removeprefix("greedy_")
        row = {"job": job_name, "pair": pair}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        rows.append(row)

    pair_order = {pair: index for index, pair in enumerate(STANDARD_PAIRS)}
    rows.sort(key=lambda r: pair_order.get(r["pair"], 99))

    columns = ["job", "pair"] + list(_SUMMARY_KEYS)
    summary_path = collected / "summary.tsv"
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for image in sorted((sweep_dir / row["job"]).glob("*.png")):
            suffix = image.stem.split("_")[-1]
            shutil.copy2(image, views_dir / f"{row['pair']}_{suffix}.png")
            view_count += 1

    mean_row = {column: _mean([r[column] for r in rows]) for column in _SUMMARY_KEYS}

    print(f"{'job':<24} {'pair':<16} {'candidates':>10} {'patches':>8} "
          f"{'f_iou':>8} {'coverage':>8} {'spill':>8}")
    print("-" * 90)
    for row in rows:
        print(
            f"{row['job']:<24} {row['pair']:<16} {row['n_candidates']:>10} "
            f"{row['final_patches']:>8} {row['final_mean_iou']:>8} "
            f"{row['final_mean_coverage']:>8} {row['final_mean_spill']:>8}"
        )
    print("-" * 90)
    print(
        f"{'mean':<24} {'':<16} {mean_row['n_candidates']:>10} "
        f"{mean_row['final_patches']:>8} {mean_row['final_mean_iou']:>8} "
        f"{mean_row['final_mean_coverage']:>8} {mean_row['final_mean_spill']:>8}"
    )

    print(f"\nWritten to {collected}:")
    print(f"  {summary_path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(STANDARD_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"greedy_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = GREEDY_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    print(f"[Sweep] greedy/{sweep_name}: {len(jobs)} job(s) on {', '.join(args.pairs)}")
    print(f"[Sweep] panel_half_size={args.panel_half_size:g}, weights="
          f"{args.view1_weight:g}/{args.view2_weight:g}, spill_weight={args.spill_weight:g}, "
          f"min_gain={args.min_gain:g}, max_panels={args.max_panels}, CPU-only ({args.partition})")

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


if __name__ == "__main__":
    main()
