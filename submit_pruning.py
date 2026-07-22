"""Submit the pruning-path comparison to SLURM.

One question: does walking the IoU-vs-pieces curve after a run beat picking a
piece-count penalty up front? Three arms, same full ``srd`` method underneath,
differing only in how the piece count is controlled:

    prune    lambda_count 1e-4 (weak enough not to bite) plus --prune-path,
             so the run grows freely and is then pruned to the knee -- the
             fewest pieces still within --prune-tolerance of the best mean IoU
             on the path. This is the recommended setting.
    penalty  lambda_count 1e-2, the knee of the earlier lambda sweep, applied
             as a fixed penalty during the run with no pruning afterwards.
    nopen    lambda_count 0: SRD with the piece-count term switched off
             entirely, the uncontrolled baseline both others are measured
             against.

Ten pairs -- the five basic ones (a detailed shape against a simple primitive)
and the five hard ones (two detailed silhouettes competing for the same
geometry) -- x three arms x one seed = 30 jobs. Everything lives under
``results/pruning/``.

Every job carries the same fixed configuration as the hard and piece-count
batches: fan-fill triangulation, swept-volume resolution 256, 100% of SRD
additions drawn from the swept volume, the planar overlap test with its hard
repair pass, and a silhouette:negative-space weight ratio of 2 at a fixed sum
of 4.5 (-> 3.0/1.5). Runs stop when both mean IoU and loss plateau, never
before --min-steps (1000) and never past --steps (4000) or --max-hours.

``run_final.py`` records step, mean IoU and piece count at every evaluation in
``history.csv`` -- the piece-count-over-time trace this batch is for -- writes
the full deletion curve to ``pruning_path.csv`` on the prune arm, and exports
the two final rendered camera views per job.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_pruning.py --dry-run

    # Submit 10 pairs x 3 arms x 1 seed = 30 jobs
    python submit_pruning.py

    # After jobs finish: aggregate into pruning/<name>/collected/
    python submit_pruning.py --sweep-name <name> --collect
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

PRUNING_ROOT = _PROJECT_ROOT / "results" / "pruning"

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

# How each arm controls the piece count. Same underlying method ('srd') in all
# three; only lambda_count and the post-run pruning differ.
ARM_SETTINGS: dict[str, dict] = {
    "prune": {"lambda_count": 1e-4, "prune_path": True},
    "penalty": {"lambda_count": 1e-2, "prune_path": False},
    "nopen": {"lambda_count": 0.0, "prune_path": False},
}
ARMS = tuple(ARM_SETTINGS)

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
        description="Submit the pruning-path comparison to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory under results/pruning/ (default: pp_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--arms",
        nargs="*",
        choices=sorted(ARMS),
        default=sorted(ARMS),
        help="Piece-count control arms to compare.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Single trial seed per config.")
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

    prune = parser.add_argument_group("pruning path (prune arm only)")
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
        for arm in args.arms:
            settings = ARM_SETTINGS[arm]
            job_name = f"pp_{pair}_{arm}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", BASE_ARM,
                "--lambda-count", f"{settings['lambda_count']:g}",
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
                "lambda": settings["lambda_count"],
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
    lines = ["job_id\tjob_name\tpair\tarm\tlambda\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['arm']}\t"
            f"{row['lambda']:g}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "lambda_count", "lambda_mode", "seed", "overlap_mode", "stop_reason",
    "final_step", "final_loss",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_overlap", "srd_total_adds", "srd_total_deletes", "total_seconds",
    # Present on the prune arm only.
    "pruned_patches", "pruned_mean_iou", "pruned_loss",
    "pruned_mean_coverage", "pruned_mean_precision", "pruned_mean_spill",
    "pruned_threshold", "pruned_target_reached",
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


def _pair_and_arm(job_name: str) -> tuple[str, str]:
    """Split 'pp_<pair>_<arm>' back into its parts."""
    stripped = re.sub(r"^pp_", "", job_name)
    for arm in ARMS:
        if stripped.endswith(f"_{arm}"):
            return stripped[: -len(arm) - 1], arm
    return stripped, "-"


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
        pair, arm = _pair_and_arm(report.parent.name)
        row = {"job": report.parent.name, "pair": pair, "arm": arm}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        # The prune arm's headline piece count is the pruned one; for the other
        # arms the run's own final count is what the method delivered. One
        # column so the arms are directly comparable.
        row["kept_patches"] = (
            row["pruned_patches"] if row["pruned_patches"] != "-" else row["final_patches"]
        )
        row["kept_mean_iou"] = (
            row["pruned_mean_iou"] if row["pruned_mean_iou"] != "-" else row["final_mean_iou"]
        )
        rows.append(row)

    group_order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    rows.sort(key=lambda r: (group_order.get(r["pair"], 99), r["arm"]))

    # --- per-job summary
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm", "kept_patches", "kept_mean_iou"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- every metric over steps, every job: the piece-count-over-time trace
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

    # --- the deletion curves from the prune arm, one long table
    paths_path = collected / "pruning_paths.csv"
    path_lines: list[str] = []
    for row in rows:
        path_csv = sweep_dir / row["job"] / "pruning_path.csv"
        if not path_csv.exists():
            continue
        body = path_csv.read_text(encoding="utf-8").splitlines()
        if not body:
            continue
        if not path_lines:
            path_lines.append(f"pair,arm,{body[0]}")
        for line in body[1:]:
            if line.strip():
                path_lines.append(f"{row['pair']},{row['arm']},{line}")
    if path_lines:
        paths_path.write_text("\n".join(path_lines) + "\n", encoding="utf-8")

    # --- the final camera views, gathered and renamed by pair and arm
    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    view_count = 0
    for row in rows:
        for view in ("view1", "view2"):
            matches = sorted((sweep_dir / row["job"]).glob(f"*_{view}.png"))
            if not matches:
                continue
            shutil.copy2(matches[0], views_dir / f"{row['pair']}_{row['arm']}_{view}.png")
            view_count += 1

    # --- headline: what each arm delivered, averaged within pair group
    by_arm_path = collected / "by_arm.tsv"
    mean_columns = [
        "kept_patches", "kept_mean_iou", "final_patches", "final_mean_iou",
        "best_mean_iou", "max_patches", "final_loss", "final_mean_spill",
        "srd_total_adds", "srd_total_deletes", "final_step", "total_seconds",
    ]
    lines = ["group\tarm\tn\t" + "\t".join(mean_columns)]
    for label, members_of in (
        ("basic", set(BASIC_PAIRS)),
        ("hard", set(HARD_PAIRS)),
        ("all", set(ALL_PAIRS)),
    ):
        for arm in sorted(ARMS):
            members = [r for r in rows if r["arm"] == arm and r["pair"] in members_of]
            if not members:
                continue
            values = [_mean([m[column] for m in members]) for column in mean_columns]
            lines.append(f"{label}\t{arm}\t{len(members)}\t" + "\t".join(values))
    by_arm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"{'job':<32} {'pair':<16} {'arm':<8} {'kept':>5} {'kept_iou':>9} "
          f"{'f_pc':>5} {'max_pc':>7} {'f_iou':>8} {'step':>6} {'stop':<10}")
    print("-" * 116)
    for row in rows:
        print(
            f"{row['job']:<32} {row['pair']:<16} {row['arm']:<8} "
            f"{row['kept_patches']:>5} {row['kept_mean_iou']:>9} "
            f"{row['final_patches']:>5} {row['max_patches']:>7} "
            f"{row['final_mean_iou']:>8} {row['final_step']:>6} {row['stop_reason']:<10}"
        )

    print("\nBy arm -- pieces kept vs IoU held:")
    print(f"{'group':<7} {'arm':<8} {'n':>3} {'kept_pc':>8} {'kept_iou':>9} "
          f"{'f_iou':>8} {'max_pc':>7} {'step':>7} {'seconds':>10}")
    for line in lines[1:]:
        fields = line.split("\t")
        group, arm, n = fields[0], fields[1], fields[2]
        values = dict(zip(mean_columns, fields[3:]))
        print(
            f"{group:<7} {arm:<8} {n:>3} {values['kept_patches']:>8} "
            f"{values['kept_mean_iou']:>9} {values['final_mean_iou']:>8} "
            f"{values['max_patches']:>7} {values['final_step']:>7} "
            f"{values['total_seconds']:>10}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_arm_path):
        print(f"  {path.name}")
    if path_lines:
        print(f"  {paths_path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"pp_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = PRUNING_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    print(
        f"[Sweep] pruning/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.arms)} arm(s) x 1 seed)"
    )
    for arm in args.arms:
        settings = ARM_SETTINGS[arm]
        print(
            f"[Sweep]   {arm:<8} lambda_count={settings['lambda_count']:g}, "
            f"prune_path={settings['prune_path']}"
        )
    print(
        f"[Sweep] base arm={BASE_ARM}, weights: silhouette={silhouette:g}, "
        f"negative_space={negative_space:g} (ratio {args.weight_ratio:g}), "
        f"steps {args.min_steps}..{args.steps} with early stop"
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
    print(f"[Sweep] manifest written to {manifest_path}")
    if not args.dry_run:
        print("Monitor with: squeue --me | grep pp_")
        print(f"Aggregate when done: python submit_pruning.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
