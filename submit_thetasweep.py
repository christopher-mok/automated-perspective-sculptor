"""Submit the camera edge-on rotation-constraint A/B to SLURM.

Every patch's orientation theta is normally projected to stay at least 15
degrees away from each camera yaw -- at initialization (``_sample_allowed_theta``
draws from the bands between the exclusion zones) and after every optimizer
step (``constrain_theta_to_camera_band`` in ``_post_step_constraints``). The
rule keeps a piece from turning edge-on to a view, where it collapses to a thin
line and contributes nothing to that silhouette. It also removes orientations
the fit might otherwise want, so it is not obviously free: this batch measures
what it costs, or buys, in IoU.

``run_final.py --unconstrained-theta`` sets that margin to 0, which disables the
rule through the existing code paths at both init and every step, so a piece may
orient edge-on to a camera. The flag is off by default and nothing else moves,
so the ``constrained`` arm reproduces the standard configuration exactly and the
only difference between the arms is the constraint.

Two arms per pair:

    constrained     the default 15-degree margin (unchanged behaviour).
    unconstrained   --unconstrained-theta, margin 0, theta free.

The configuration is otherwise the IoU-measurement setup the areaweights batch
used and is the point of contact with it: the full SRD arm at the selected
0.25 silhouette:negative-space ratio under area normalization, planar overlap
with its repair pass, swept-volume resolution 256, and lambda_count 0 so nothing
rations piece count -- the fit is free to grow to its IoU ceiling, which is the
quantity being compared. IoU, not piece count, is the tracked outcome here.

Because the constraint reshapes the random theta initialization, a single seed
would confound "the constraint changed the answer" with "the two arms drew
different starting layouts". The batch runs three seeds per arm and compares
means, so the reported delta is the constraint's effect rather than one draw's.

    7 pairs x 2 arms x 3 seeds = 42 jobs, all under results/thetasweep/.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_thetasweep.py --dry-run

    # 7 pairs x 2 arms x 3 seeds = 42 jobs
    python submit_thetasweep.py

    # After jobs finish
    python submit_thetasweep.py --sweep-name <name> --collect
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

THETASWEEP_ROOT = _PROJECT_ROOT / "results" / "thetasweep"

# One detailed shape against a simple primitive.
BASIC_PAIRS = (
    "sun_moon",
    "cat_face_bass",
    "horse_circle",
    "acm_scf",
)

# Two detailed silhouettes competing for the same geometry.
HARD_PAIRS = (
    "teapot_droplets",
    "water_fire",
    "dance_argument",
)

ALL_PAIRS = BASIC_PAIRS + HARD_PAIRS

BASE_ARM = "srd"

# silhouette : negative_space at a fixed sum, under --area-normalized-view-loss.
# The 0.25 the areaweights sweep selected; the constraint A/B holds it fixed.
WEIGHT_RATIO = 0.25
WEIGHT_SUM = 4.5

# Piece count is not rationed here, so the fit reaches its IoU ceiling.
LAMBDA_COUNT = 0.0

# The A and B arms. The extra flags each arm passes to run_final.py; the
# constrained arm passes nothing, so it is the untouched default.
ARM_FLAGS = {
    "constrained": [],
    "unconstrained": ["--unconstrained-theta"],
}
ARMS = tuple(ARM_FLAGS)

DEFAULT_SEEDS = (0, 1, 2)

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
        description="Submit the edge-on rotation-constraint A/B to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Batch directory under results/thetasweep/ (default: th_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument("--arms", nargs="*", default=None, help="Override the arm list.")
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS),
        help="Trial seeds; the batch averages IoU over them.",
    )
    parser.add_argument("--weight-ratio", type=float, default=WEIGHT_RATIO)
    parser.add_argument("--weight-sum", type=float, default=WEIGHT_SUM)

    parser.add_argument("--steps", type=int, default=3000, help="Hard ceiling.")
    parser.add_argument("--min-steps", type=int, default=1000)
    parser.add_argument("--patience-steps", type=int, default=300)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--n-patches", type=int, default=20)
    parser.add_argument("--srd-candidates", type=int, default=32)
    parser.add_argument("--lambda-count", type=float, default=LAMBDA_COUNT)
    parser.add_argument("--swept-resolution", type=int, default=256)
    parser.add_argument("--swept-spawn-fraction", type=float, default=1.0)
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
        for arm in args.arms:
            for seed in args.seeds:
                job_name = f"th_{pair}_{arm}_s{seed}"
                output_dir = sweep_dir / job_name
                command = [
                    args.python, "run_final.py",
                    "--target1", str(image_dir / target1),
                    "--target2", str(image_dir / target2),
                    "--arm", BASE_ARM,
                    "--srd-candidates", str(args.srd_candidates),
                    "--lambda-count", f"{args.lambda_count:g}",
                    *ARM_FLAGS[arm],
                    "--overlap-mode", "planar",
                    "--seed", str(seed),
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
                    "--area-normalized-view-loss",
                    "--render-scale", str(args.render_scale),
                    "--max-hours", str(args.max_hours),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ]
                jobs.append({
                    "name": job_name,
                    "pair": pair,
                    "arm": arm,
                    "seed": seed,
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
        ["sbatch", str(script_path)], capture_output=True, text=True, check=True
    )
    match = re.search(r"(\d+)", result.stdout)
    return match.group(1) if match else result.stdout.strip()


def _write_manifest(sweep_dir: Path, rows: list[dict]) -> Path:
    manifest_path = sweep_dir / "manifest.tsv"
    lines = ["job_id\tjob_name\tpair\tarm\tseed\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t"
            f"{row['arm']}\t{row['seed']}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "unconstrained_theta", "silhouette_weight", "negative_space_weight",
    "weight_ratio", "area_normalized_view_loss", "lambda_count", "seed",
    "overlap_mode", "stop_reason", "final_step", "final_loss",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_overlap", "srd_total_adds", "srd_total_splits", "srd_total_deletes",
    "total_seconds",
)


def _grab(text: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "-"


def _nums(values: list[str]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number < 0:
            continue
        out.append(number)
    return out


def _mean(values: list[str]) -> str:
    numbers = _nums(values)
    return f"{sum(numbers) / len(numbers):.6g}" if numbers else "-"


def _pair_and_arm_and_seed(job_name: str) -> tuple[str, str, str]:
    """Split 'th_<pair>_<arm>_s<seed>' back into its parts."""
    stripped = re.sub(r"^th_", "", job_name)
    seed_match = re.search(r"_s(\d+)$", stripped)
    seed = seed_match.group(1) if seed_match else "-"
    if seed_match:
        stripped = stripped[: seed_match.start()]
    for arm in ARMS:
        if stripped.endswith(f"_{arm}"):
            return stripped[: -len(arm) - 1], arm, seed
    return stripped, "-", seed


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
        pair, arm, seed = _pair_and_arm_and_seed(report.parent.name)
        row = {"job": report.parent.name, "pair": pair, "arm": arm, "seed": seed}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        rows.append(row)

    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    group_order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    rows.sort(key=lambda r: (group_order.get(r["pair"], 99),
                             arm_order.get(r["arm"], 99), r["seed"]))

    # --- per-job summary
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm", "seed"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "-")) for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- IoU / piece-count curves, tagged by arm and seed
    curves_path = collected / "curves.csv"
    curve_lines: list[str] = []
    for row in rows:
        history = sweep_dir / row["job"] / "history.csv"
        if not history.exists():
            continue
        body = history.read_text(encoding="utf-8").splitlines()
        if not body:
            continue
        prefix = f"{row['pair']},{row['arm']},{row['seed']}"
        if not curve_lines:
            curve_lines.append(f"pair,arm,seed,{body[0]}")
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{prefix},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    # --- final renders, renamed by pair/arm/seed
    views_dir = collected / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for image in sorted((sweep_dir / row["job"]).glob("*.png")):
            suffix = image.stem.split("_")[-1]
            shutil.copy2(
                image, views_dir / f"{row['pair']}_{row['arm']}_s{row['seed']}_{suffix}.png"
            )

    # --- per-pair A/B: mean over seeds, and the constrained->unconstrained delta
    metric = "best_mean_iou"
    by_pair_path = collected / "by_pair.tsv"
    pair_lines = ["pair\tgroup\tn_seeds\tconstrained_iou\tunconstrained_iou\tdelta_iou"]
    per_pair_delta: dict[str, float] = {}
    for pair in ALL_PAIRS:
        con = _mean([r[metric] for r in rows if r["pair"] == pair and r["arm"] == "constrained"])
        unc = _mean([r[metric] for r in rows if r["pair"] == pair and r["arm"] == "unconstrained"])
        n = len([r for r in rows if r["pair"] == pair and r["arm"] == "constrained"])
        group = "basic" if pair in BASIC_PAIRS else "hard"
        if con != "-" and unc != "-":
            delta = float(unc) - float(con)
            per_pair_delta[pair] = delta
            delta_s = f"{delta:+.6g}"
        else:
            delta_s = "-"
        pair_lines.append(f"{pair}\t{group}\t{n}\t{con}\t{unc}\t{delta_s}")
    by_pair_path.write_text("\n".join(pair_lines) + "\n", encoding="utf-8")

    # --- group-level A/B on both final and best IoU
    by_arm_path = collected / "by_arm.tsv"
    mean_columns = [
        "final_mean_iou", "best_mean_iou", "final_view1_iou", "final_view2_iou",
        "final_patches", "max_patches", "final_mean_spill", "final_mean_coverage",
        "final_mean_precision", "final_loss", "final_step", "total_seconds",
    ]
    arm_lines = ["group\tarm\tn\t" + "\t".join(mean_columns)]
    for label, members_of in (
        ("basic", set(BASIC_PAIRS)),
        ("hard", set(HARD_PAIRS)),
        ("all", set(ALL_PAIRS)),
    ):
        for arm in ARMS:
            members = [r for r in rows if r["arm"] == arm and r["pair"] in members_of]
            if not members:
                continue
            values = [_mean([m[column] for m in members]) for column in mean_columns]
            arm_lines.append(f"{label}\t{arm}\t{len(members)}\t" + "\t".join(values))
    by_arm_path.write_text("\n".join(arm_lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"{'pair':<16} {'group':<6} {'constrained':>12} {'unconstrained':>14} {'delta':>9}")
    print("-" * 62)
    for line in pair_lines[1:]:
        pair, group, n, con, unc, delta = line.split("\t")
        print(f"{pair:<16} {group:<6} {con:>12} {unc:>14} {delta:>9}")
    print("\nGroup means (best_mean_iou over seeds):")
    for label, members_of in (("basic", set(BASIC_PAIRS)),
                              ("hard", set(HARD_PAIRS)),
                              ("all", set(ALL_PAIRS))):
        con = _mean([r["best_mean_iou"] for r in rows
                     if r["arm"] == "constrained" and r["pair"] in members_of])
        unc = _mean([r["best_mean_iou"] for r in rows
                     if r["arm"] == "unconstrained" and r["pair"] in members_of])
        if con != "-" and unc != "-":
            print(f"  {label:<6} constrained={con}  unconstrained={unc}  "
                  f"delta={float(unc) - float(con):+.4f}")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_pair_path, by_arm_path):
        print(f"  {path.name}")


def main() -> None:
    args = _parse_args()
    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    if args.arms is None:
        args.arms = list(ARMS)

    sweep_name = args.sweep_name or f"th_{datetime.now():%Y%m%d_%H%M}"
    sweep_dir = THETASWEEP_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)

    for job in jobs:
        script_path = _write_sbatch_script(job, args, sweep_dir)
        if args.dry_run:
            print(f"[Dry run] wrote {script_path}")
        else:
            job["job_id"] = _submit(script_path)
            print(f"[Submitted] {job['job_id']}: {job['name']}")

    _write_manifest(sweep_dir, jobs)
    print(f"[Batch] manifest written to {sweep_dir / 'manifest.tsv'}")
    if not args.dry_run:
        print("Monitor with: squeue --me | grep th_")
        print(f"Aggregate when done: python submit_thetasweep.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
