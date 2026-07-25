"""Submit the hinge2 / free / prune / no-SRD comparison to SLURM.

Four ways of arriving at a piece count, on the seven pairs the hinge2 batch
runs on, under one configuration so the only difference is the mechanism:

    hinge2   The repaired hinge at its best setting from h2_20260723_0259:
             count_power 1 with the hard floor, the fit-then-shed schedule,
             delete_eval_steps 15, shed_patience_evals 8, count_lambda 500.
             Pieces are shed *during* the run, by rewrites scored on
             J = n + lambda * hinge(target - iou) - mu * hinge(iou - target).
             lambda 500 is the arm that produced both of that batch's real
             sheds (cat_face_bass 20 -> 8 at 0.917, water_fire 20 -> 12 at
             0.922); l200 and l2000 held target but shed little or nothing.

    free     Unconstrained SRD: --arm srd with lambda_count 0 and no count
             objective, so adds, splits and deletions are scored on the image
             loss alone and nothing prices a piece. It is the other end of the
             bid from base -- what the rewrites do when they are allowed to
             spend as many pieces as the loss will pay for -- and it is the
             configuration the areaweights sweep measured the 0.25 ratio
             under, so its numbers are directly comparable to that sweep.

    prune    The pcompare prune arm: no count pressure during the run
             (lambda_count 1e-4, weak enough not to bite), then a post-hoc
             pruning path that deletes the least-damaging piece repeatedly,
             records mean IoU at every piece count, and leaves the model at
             the knee. Same destination, reached by search after the fit
             rather than by pressure during it, and it is the arm that set the
             IoU ceiling those pairs can reach (0.944 / 0.964).

    base     --arm basic: no SRD at all. Fixed 20 pieces, no adds, splits or
             deletions, so it is the floor both mechanisms are bidding
             against -- what the same 20 pieces and the same gradient steps do
             when nothing rewrites them.

Every arm runs at the area weights the areaweights sweep selected --
silhouette : negative_space = 0.25 at a fixed sum of 4.5, i.e. 0.9 / 3.6, with
--area-normalized-view-loss -- rather than the 2.0 without normalization the
h2 batch used. See submit_hinge2.py's WEIGHT_RATIO comment for the measurement
and its two caveats. That makes this batch's numbers comparable to each other
and to a re-run h2 batch, but *not* to h2_20260723_0259 or the pcompare
batches, which were fit under a differently scaled loss.

Instrumentation, which is the other reason this batch exists:

  * --eval-interval 10 rather than 25, so history.csv carries all 30 tracked
    metrics -- loss, per-view and mean IoU, spill, precision, coverage, the
    weighted silhouette and negative-space terms, piece count, overlap,
    lambda, cumulative adds/splits/deletes, and the shed-phase flags -- at 10
    step resolution for every arm.
  * --render-every 10, writing renders/step<NNNNN>_view{1,2}.png at the
    optimization resolution, so every row of history.csv has the geometry that
    produced it. A 3000 step run leaves ~300 pairs, a few MB.

The extra render per 10 steps sits inside the timed region, so total_seconds
is comparable within this batch but runs slower than the uninstrumented
batches; wall-clock is not the measurement here.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_h2compare.py --dry-run

    # 7 pairs x 4 arms x 1 seed = 28 jobs
    python submit_h2compare.py

    # Just the two mechanism arms: 7 pairs x 2 arms = 14 jobs
    python submit_h2compare.py --arms free hinge2

    # After jobs finish
    python submit_h2compare.py --sweep-name <name> --collect
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
from submit_hinge2 import (
    ALL_PAIRS,
    AREA_NORMALIZED_VIEW_LOSS,
    FULL_STACK,
    INFEASIBLE_PAIRS,
    WEIGHT_RATIO,
    WEIGHT_SUM,
)

_PROJECT_ROOT = Path(__file__).resolve().parent

H2COMPARE_ROOT = _PROJECT_ROOT / "results" / "h2compare"

# The count_lambda the hinge2 arm runs at: 1/500 = 0.002 IoU is what a piece
# may cost. See the module docstring for why this one and not l200 / l2000.
HINGE2_LAMBDA = "500"

# base_arm is run_final.py's --arm ('basic' switches SRD off entirely);
# everything else is the flags that distinguish the mechanism.
ARM_SETTINGS: dict[str, dict] = {
    "hinge2": {
        "base_arm": "srd",
        "count_objective": True,
        "prune_path": False,
        "lambda_count": 0.0,
    },
    "free": {
        "base_arm": "srd",
        "count_objective": False,
        "prune_path": False,
        "lambda_count": 0.0,
    },
    "prune": {
        "base_arm": "srd",
        "count_objective": False,
        "prune_path": True,
        "lambda_count": 1e-4,
    },
    "base": {
        "base_arm": "basic",
        "count_objective": False,
        "prune_path": False,
        "lambda_count": 0.0,
    },
}
ARMS = tuple(ARM_SETTINGS)

SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --gres={gres}
#SBATCH --chdir={project_root}
#SBATCH --output={log_dir}/%x.%j.out
#SBATCH --error={log_dir}/%x.%j.err

set +eu
{env_setup}
set -eu

ulimit -n 50000

{command}
"""


def _weights(ratio: float, weight_sum: float) -> tuple[float, float]:
    """Split weight_sum into (silhouette, negative_space) at the given ratio."""
    negative_space = weight_sum / (1.0 + ratio)
    return weight_sum - negative_space, negative_space


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the hinge2 / prune / no-SRD comparison to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Batch directory under results/h2compare/ (default: h2c_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--arms",
        nargs="*",
        choices=sorted(ARMS),
        default=list(ARMS),
        help="Override the arm list.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single trial seed per pair.")

    tracking = parser.add_argument_group("tracking")
    tracking.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help="Sample every tracked metric into history.csv every N steps.",
    )
    tracking.add_argument(
        "--render-every",
        type=int,
        default=10,
        help="Export both camera views every N steps into renders/.",
    )
    tracking.add_argument(
        "--render-every-scale",
        type=int,
        default=1,
        help="Resolution multiplier for those snapshots (1 = optimization "
             "resolution, a few tens of KB per pair).",
    )

    hinge = parser.add_argument_group("hinge2 arm")
    hinge.add_argument("--count-target-iou", type=float, default=0.91)
    hinge.add_argument("--count-lambda", default=HINGE2_LAMBDA)

    prune = parser.add_argument_group("prune arm")
    prune.add_argument("--prune-min-pieces", type=int, default=1)
    prune.add_argument(
        "--prune-refit-steps",
        type=int,
        default=4,
        help="Gradient steps re-fitting the survivors after each deletion.",
    )
    prune.add_argument(
        "--prune-tolerance",
        type=float,
        default=0.01,
        help="Keep the fewest pieces within this much mean IoU of the best "
             "point on the path.",
    )

    parser.add_argument("--steps", type=int, default=3000, help="Hard step ceiling.")
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--patience-steps", type=int, default=300)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--swept-spawn-fraction", type=float, default=1.0)
    parser.add_argument("--weight-ratio", type=float, default=WEIGHT_RATIO)
    parser.add_argument("--weight-sum", type=float, default=WEIGHT_SUM)
    parser.add_argument("--render-scale", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=23.0)
    parser.add_argument("--device", default="cuda")

    cluster = parser.add_argument_group("cluster resources")
    cluster.add_argument("--partition", default="3090-gcondo")
    cluster.add_argument("--gres", default="gpu:1")
    cluster.add_argument("--time", default="24:00:00", help="SLURM walltime (> --max-hours).")
    cluster.add_argument("--mem", default="125G")
    cluster.add_argument("--cpus", type=int, default=6)
    cluster.add_argument("--python", default="python")
    cluster.add_argument(
        "--env-setup",
        default="source /oscar/home/cjmok/.bashrc\nconda activate myenv",
    )

    mode = parser.add_argument_group("actions")
    mode.add_argument("--dry-run", action="store_true", help="Write scripts, submit nothing.")
    mode.add_argument("--collect", action="store_true", help="Aggregate results and exit.")
    return parser.parse_args()


def _hinge2_flags(count_lambda: str) -> list[str]:
    """FULL_STACK from submit_hinge2, at this batch's lambda, as CLI flags."""
    settings = {**FULL_STACK, "count_lambda": count_lambda}
    flags: list[str] = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        else:
            flags.extend([flag, str(value)])
    return flags


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for arm in args.arms:
            settings = ARM_SETTINGS[arm]
            job_name = f"h2c_{pair}_{arm}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", settings["base_arm"],
                "--lambda-count", f"{settings['lambda_count']:g}",
                "--overlap-mode", "planar",
                "--seed", str(args.seed),
                "--steps", str(args.steps),
                "--early-stop",
                "--min-steps", str(args.min_steps),
                "--patience-steps", str(args.patience_steps),
                "--eval-interval", str(args.eval_interval),
                "--render-every", str(args.render_every),
                "--render-every-scale", str(args.render_every_scale),
                "--n-patches", str(args.n_patches),
                "--swept-resolution", str(args.swept_resolution),
                "--swept-spawn-fraction", str(args.swept_spawn_fraction),
                "--silhouette-weight", f"{silhouette:g}",
                "--negative-space-weight", f"{negative_space:g}",
                *(["--area-normalized-view-loss"] if AREA_NORMALIZED_VIEW_LOSS else []),
                "--render-scale", str(args.render_scale),
                "--max-hours", str(args.max_hours),
                "--device", args.device,
                "--output-dir", str(output_dir),
            ]
            if settings["count_objective"]:
                command += [
                    "--count-objective",
                    "--count-target-iou", f"{args.count_target_iou:g}",
                    *_hinge2_flags(args.count_lambda),
                ]
            if settings["prune_path"]:
                command += [
                    "--prune-path",
                    "--prune-min-pieces", str(args.prune_min_pieces),
                    "--prune-refit-steps", str(args.prune_refit_steps),
                    "--prune-tolerance", str(args.prune_tolerance),
                ]
            jobs.append({
                "name": job_name,
                "pair": pair,
                "arm": arm,
                "output_dir": output_dir,
                "command": command,
            })
    return jobs


def _write_sbatch_script(job: dict, args: argparse.Namespace, sweep_dir: Path) -> Path:
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = SBATCH_TEMPLATE.format(
        job_name=job["name"],
        partition=args.partition,
        gres=args.gres,
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
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\tarm\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{row['arm']}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    # NB: not "arm" -- the report's arm= is run_final.py's base arm
    # ('basic'/'srd'), while this batch's arm is the mechanism name taken from
    # the job directory. Pulling both under one key would silently overwrite it.
    "base_arm", "seed", "silhouette_weight", "negative_space_weight", "weight_ratio",
    "area_normalized_view_loss", "eval_interval", "render_every",
    "count_objective", "count_lambda", "lambda_count",
    "stop_reason", "final_step",
    "count_target_reached", "count_final_deficit",
    "final_mean_iou", "best_mean_iou", "final_patches",
    "shed_entered_step", "shed_entry_patches", "shed_evals",
    "min_iou_after_target",
    "pruned_patches", "pruned_mean_iou", "pruned_target_reached",
    "start_patches", "max_patches", "min_patches",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_overlap", "srd_total_adds", "srd_total_splits", "srd_total_deletes",
    "total_seconds",
)


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "-"


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
        name = re.sub(r"^h2c_", "", report.parent.name)
        arm = name.rsplit("_", 1)[1]
        pair = name.rsplit("_", 1)[0]
        row = {"job": report.parent.name, "pair": pair, "arm": arm}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        row["base_arm"] = _grab(text, "arm")
        # Frame series length, so a truncated run is visible in the summary
        # rather than only on disk.
        frames = sweep_dir / row["job"] / "renders"
        row["frames"] = str(len(list(frames.glob("step*_view1.png"))) if frames.is_dir() else 0)
        rows.append(row)

    pair_order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    rows.sort(key=lambda r: (pair_order.get(r["pair"], 99), arm_order.get(r["arm"], 99)))

    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm", "frames"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Every metric of every arm on one axis, tagged by pair and arm.
    curves_path = collected / "curves.csv"
    curve_lines: list[str] = []
    missing_curves = 0
    for row in rows:
        history = sweep_dir / row["job"] / "history.csv"
        if not history.exists():
            missing_curves += 1
            continue
        body = history.read_text(encoding="utf-8").splitlines()
        if not body:
            missing_curves += 1
            continue
        if not curve_lines:
            curve_lines.append(f"pair,arm,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{row['pair']},{row['arm']},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # The prune arm's IoU-vs-pieces curves, which have no counterpart in the
    # other two arms and so cannot go in curves.csv.
    prune_path = collected / "pruning_paths.csv"
    prune_lines: list[str] = []
    for row in rows:
        path = sweep_dir / row["job"] / "pruning_path.csv"
        if not path.exists():
            continue
        body = path.read_text(encoding="utf-8").splitlines()
        if not body:
            continue
        if not prune_lines:
            prune_lines.append(f"pair,arm,{body[0]}")
        for line in body[1:]:
            if line.strip():
                prune_lines.append(f"{row['pair']},{row['arm']},{line}")
    if prune_lines:
        prune_path.write_text("\n".join(prune_lines) + "\n", encoding="utf-8")

    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for view in ("view1", "view2"):
            matches = sorted(
                p for p in (sweep_dir / row["job"]).glob(f"*_{view}.png")
            )
            if not matches:
                continue
            shutil.copy2(matches[0], views_dir / f"{row['pair']}_{row['arm']}_{view}.png")
            view_count += 1

    print(f"{'pair':<18} {'arm':<7} {'f_iou':>8} {'best':>8} {'pcs':>4} "
          f"{'held':>5} {'shed@':>6} {'prune_pc':>8} {'frames':>6} {'step':>5} {'stop':<15}")
    print("-" * 104)
    for row in rows:
        print(
            f"{row['pair']:<18} {row['arm']:<7} {row['final_mean_iou']:>8} "
            f"{row['best_mean_iou']:>8} {row['final_patches']:>4} "
            f"{row['count_target_reached']:>5} {row['shed_entered_step']:>6} "
            f"{row['pruned_patches']:>8} {row['frames']:>6} "
            f"{row['final_step']:>5} {row['stop_reason']:<15}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path):
        print(f"  {path.name}")
    if prune_lines:
        print(f"  {prune_path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"h2c_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = H2COMPARE_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)

    print(
        f"[Batch] h2compare/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.arms)} arm(s) x 1 seed)"
    )
    descriptions = {
        "hinge2": f"srd + full repaired stack, lambda={args.count_lambda}, "
                  f"target={args.count_target_iou:g} (sheds during the run)",
        "free": "srd, lambda_count=0, no count objective (unconstrained rewrites)",
        "prune": f"srd, lambda_count=1e-4, --prune-path "
                 f"(sheds after the run, tolerance {args.prune_tolerance:g})",
        "base": "--arm basic: no SRD, fixed piece count",
    }
    for arm in args.arms:
        print(f"[Batch]   {arm:<7} {descriptions[arm]}")
    print(
        f"[Batch] weights: silhouette={silhouette:g}, negative_space={negative_space:g} "
        f"(ratio {args.weight_ratio:g}, area_normalized={AREA_NORMALIZED_VIEW_LOSS}) "
        f"-- selected by the areaweights sweep, so NOT comparable to "
        f"h2_20260723_0259 or the pcompare batches"
    )
    print(
        f"[Batch] tracking: all metrics every {args.eval_interval} steps, "
        f"views every {args.render_every} steps at scale "
        f"{args.render_every_scale} (~{args.steps // max(1, args.render_every)} "
        f"pairs per run at the step ceiling)"
    )
    print(
        f"[Batch] n_patches={args.n_patches}, steps {args.min_steps}..{args.steps} "
        f"with early stop; infeasibility control: {', '.join(INFEASIBLE_PAIRS)}"
    )

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
    print(f"[Batch] manifest written to {manifest_path}")
    if not args.dry_run:
        print("Monitor with: squeue --me | grep h2c_")
        print(f"Aggregate when done: python submit_h2compare.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
