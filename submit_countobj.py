"""Submit the piece-count-objective batch to SLURM.

Every other batch minimises image loss and treats the piece count as something
to be rationed on the side, by a ``lambda_count`` penalty that charges for
pieces however good the fit already is. This one inverts that: pieces *are* the
objective, and IoU is a constraint,

    J = n_patches + count_lambda * hinge(0.90 - mean_iou)^2

with ``hinge = max(0, .)``. Above 0.90 mean IoU the second term is exactly
zero, so a rewrite is scored purely on the pieces it costs and SRD spends the
rest of the run shedding whatever the sculpture can spare; below 0.90 the term
switches back on and coverage has to be bought back first. Rewrites are scored
by the change in J instead of the change in image loss, and every piece (not
just the tiny ones) becomes a deletion candidate. The gradient steps between
rewrites still descend the ordinary image loss -- J governs the discrete
rewrites, which is where the piece count is decided.

``count_lambda`` is the one free number: it is the exchange rate between a
piece and a unit of squared IoU deficit. At the default 2000, sitting 0.02
below target costs 0.8 pieces and sitting 0.10 below costs 20, so a run grows
freely while it is far from the threshold and stops paying for IoU as it
approaches. Because the penalty is quadratic its slope vanishes at the kink, so
runs are expected to settle slightly *under* 0.90 rather than exactly on it;
``--count-power 1`` or ``--count-softplus-beta`` are the two levers if that
undershoot turns out to matter.

Ten pairs -- the five basic ones (a detailed shape against a simple primitive)
and the five hard ones (two detailed silhouettes competing for the same
geometry) -- x one arm x one seed = 10 jobs, under ``results/countobj/``.
SRD is otherwise at its defaults (interval 50, 32 candidates, the default
0.35/0.15/0.50 add/delete/split mix), and the run configuration is the one
shared with the hard, piece-count and pruning batches: fan-fill triangulation,
swept-volume resolution 256, 100% of SRD additions drawn from the swept volume,
the planar overlap test with its hard repair pass, and a silhouette:negative-
space weight ratio of 2 at a fixed sum of 4.5 (-> 3.0/1.5). Runs stop when both
mean IoU and loss plateau, never before --min-steps (1000) and never past
--steps (4000) or --max-hours, and each exports its two final camera views.

The headline numbers are in report.txt: ``patches_at_threshold`` (what it took
to first reach 0.90), ``final_patches`` (what survived the shedding after that)
and ``count_target_reached``.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_countobj.py --dry-run

    # Submit 10 pairs x 1 arm x 1 seed = 10 jobs
    python submit_countobj.py

    # After jobs finish: aggregate into countobj/<name>/collected/
    python submit_countobj.py --sweep-name <name> --collect
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

COUNTOBJ_ROOT = _PROJECT_ROOT / "results" / "countobj"

# One detailed shape against a simple primitive.
BASIC_PAIRS = (
    "sun_moon",
    "horse_circle",
    "cat_face_bass",
    "water_fire",
    "acm_scf",
)

# Two detailed silhouettes, so both views compete for the same geometry.
HARD_PAIRS = (
    "siggraph_sigchi",
    "crane_crab",
    "teapot_droplets",
    "dancer_guitar",
    "dance_argument",
)

ALL_PAIRS = BASIC_PAIRS + HARD_PAIRS

BASE_ARM = "srd"

# silhouette : negative_space = 2, holding their sum at 4.5 so the view loss
# keeps the same magnitude relative to the untouched rgb and overlap terms.
WEIGHT_RATIO = 2.0
WEIGHT_SUM = 4.5

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
        description="Submit the piece-count-objective batch to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Batch directory under results/countobj/ (default: co_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument("--seed", type=int, default=0, help="Single trial seed per pair.")

    objective = parser.add_argument_group("piece-count objective")
    objective.add_argument("--count-target-iou", type=float, default=0.90)
    objective.add_argument(
        "--count-lambda",
        type=float,
        default=2000.0,
        help="Pieces one unit of squared IoU deficit is worth.",
    )
    objective.add_argument(
        "--count-power",
        type=float,
        default=2.0,
        help="Hinge exponent; 2 is the quadratic, 1 keeps a constant slope.",
    )
    objective.add_argument(
        "--count-softplus-beta",
        type=float,
        default=0.0,
        help="0 uses the hard max; positive smooths the kink at that sharpness.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Hard ceiling; runs normally stop earlier on convergence.",
    )
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--patience-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
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


def _build_jobs(args: argparse.Namespace, sweep_dir: Path) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        job_name = f"co_{pair}"
        output_dir = sweep_dir / job_name
        command = [
            args.python, "run_final.py",
            "--target1", str(image_dir / target1),
            "--target2", str(image_dir / target2),
            "--arm", BASE_ARM,
            "--count-objective",
            "--count-target-iou", f"{args.count_target_iou:g}",
            "--count-lambda", f"{args.count_lambda:g}",
            "--count-power", f"{args.count_power:g}",
            "--count-softplus-beta", f"{args.count_softplus_beta:g}",
            "--overlap-mode", "planar",
            "--seed", str(args.seed),
            "--steps", str(args.steps),
            "--early-stop",
            "--min-steps", str(args.min_steps),
            "--patience-steps", str(args.patience_steps),
            "--eval-interval", str(args.eval_interval),
            "--n-patches", str(args.n_patches),
            "--swept-resolution", str(args.swept_resolution),
            "--swept-spawn-fraction", str(args.swept_spawn_fraction),
            "--silhouette-weight", f"{silhouette:g}",
            "--negative-space-weight", f"{negative_space:g}",
            "--render-scale", str(args.render_scale),
            "--max-hours", str(args.max_hours),
            "--device", args.device,
            "--output-dir", str(output_dir),
        ]
        jobs.append({
            "name": job_name,
            "pair": pair,
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
    lines = ["job_id\tjob_name\tpair\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "count_target_iou", "count_lambda", "count_power", "count_softplus_beta",
    "seed", "overlap_mode", "stop_reason", "final_step", "final_loss",
    # Did the constraint hold, and what did the objective come to?
    "count_target_reached", "count_final_deficit", "count_final_objective",
    "patches_at_threshold", "step_at_threshold", "patches_shed_after_threshold",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_overlap",
    "srd_total_adds", "srd_total_splits", "srd_total_deletes",
    "iou_0p90_steps", "total_seconds",
)


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "-"


def _mean(values: list[str]) -> str:
    numbers = []
    for value in values:
        try:
            number = float(value)
        except ValueError:
            continue
        if number < 0:
            continue
        numbers.append(number)
    if not numbers:
        return "-"
    return f"{sum(numbers) / len(numbers):.6g}"


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
        pair = re.sub(r"^co_", "", report.parent.name)
        row = {"job": report.parent.name, "pair": pair}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        rows.append(row)

    order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    rows.sort(key=lambda r: order.get(r["pair"], 99))

    summary_path = collected / "summary.tsv"
    columns = ["job", "pair"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- IoU and piece count over steps: the shedding phase after the
    #     threshold is crossed is the thing to look at here.
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
        group = "basic" if row["pair"] in BASIC_PAIRS else "hard"
        if not curve_lines:
            curve_lines.append(f"pair,group,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{row['pair']},{group},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for view in ("view1", "view2"):
            matches = sorted((sweep_dir / row["job"]).glob(f"*_{view}.png"))
            if not matches:
                continue
            shutil.copy2(matches[0], views_dir / f"{row['pair']}_{view}.png")
            view_count += 1

    by_group_path = collected / "by_group.tsv"
    mean_columns = [
        "final_patches", "patches_at_threshold", "patches_shed_after_threshold",
        "final_mean_iou", "best_mean_iou", "count_final_deficit",
        "count_final_objective", "max_patches", "srd_total_adds",
        "srd_total_splits", "srd_total_deletes", "final_step", "total_seconds",
    ]
    lines = ["group\tn\treached\t" + "\t".join(mean_columns)]
    for label, members_of in (
        ("basic", set(BASIC_PAIRS)),
        ("hard", set(HARD_PAIRS)),
        ("all", set(ALL_PAIRS)),
    ):
        members = [r for r in rows if r["pair"] in members_of]
        if not members:
            continue
        reached = sum(1 for m in members if m["count_target_reached"] == "True")
        values = [_mean([m[column] for m in members]) for column in mean_columns]
        lines.append(f"{label}\t{len(members)}\t{reached}\t" + "\t".join(values))
    by_group_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'pair':<18} {'reached':>8} {'f_iou':>8} {'thr_pc':>7} {'f_pc':>5} "
          f"{'shed':>5} {'adds':>6} {'splits':>7} {'dels':>6} {'step':>6} {'stop':<10}")
    print("-" * 108)
    for row in rows:
        print(
            f"{row['pair']:<18} {row['count_target_reached']:>8} "
            f"{row['final_mean_iou']:>8} {row['patches_at_threshold']:>7} "
            f"{row['final_patches']:>5} {row['patches_shed_after_threshold']:>5} "
            f"{row['srd_total_adds']:>6} {row['srd_total_splits']:>7} "
            f"{row['srd_total_deletes']:>6} {row['final_step']:>6} "
            f"{row['stop_reason']:<10}"
        )

    print("\nBy group -- pieces kept against the 0.90 IoU floor:")
    print(f"{'group':<7} {'n':>3} {'reached':>8} {'f_pc':>7} {'thr_pc':>7} "
          f"{'shed':>6} {'f_iou':>8} {'deficit':>9} {'J':>9}")
    for line in lines[1:]:
        fields = line.split("\t")
        group, n, reached = fields[0], fields[1], fields[2]
        values = dict(zip(mean_columns, fields[3:]))
        print(
            f"{group:<7} {n:>3} {reached:>8} {values['final_patches']:>7} "
            f"{values['patches_at_threshold']:>7} "
            f"{values['patches_shed_after_threshold']:>6} "
            f"{values['final_mean_iou']:>8} {values['count_final_deficit']:>9} "
            f"{values['count_final_objective']:>9}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_group_path):
        print(f"  {path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"co_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = COUNTOBJ_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    hinge = "softplus" if args.count_softplus_beta > 0 else "max"
    print(
        f"[Batch] countobj/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x 1 seed)"
    )
    print(
        f"[Batch]   J = n_patches + {args.count_lambda:g} * "
        f"{hinge}(0, {args.count_target_iou:g} - mean_iou)^{args.count_power:g}"
    )
    print(
        f"[Batch] base arm={BASE_ARM} at default SRD settings, weights: "
        f"silhouette={silhouette:g}, negative_space={negative_space:g} "
        f"(ratio {args.weight_ratio:g}), steps {args.min_steps}..{args.steps} "
        f"with early stop"
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
        print("Monitor with: squeue --me | grep co_")
        print(f"Aggregate when done: python submit_countobj.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
