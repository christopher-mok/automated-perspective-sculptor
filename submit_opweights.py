"""Submit the SRD operation-weight sweep to SLURM.

SRD divides its per-step candidate budget between three rewrite kinds -- add a
new piece, delete one, split one in two -- at a fixed 0.35/0.15/0.50. Those are
proportions of the candidates SRD *looks at*, not acceptance probabilities:
every candidate still has to beat the loss. So the mix does not decide what the
sculpture ends up being, it decides how much of each move SRD gets to try, and
therefore how fast it grows and how it grows. Two families here, each holding
the other two weights at their default ratio so one knob moves at a time:

    add      add_weight from the default 0.35 down to 0.05, with delete and
             split keeping their default 0.15:0.50 ratio over the remainder.
             The default is the *upper* bound: the question is what is lost by
             offering fewer additions, and whether splitting picks up the slack.
    split    split_weight over 0.20 .. 0.80 with the default 0.50 in the middle,
             add and delete keeping their default 0.35:0.15 (7:3) ratio over the
             remainder. Splits subdivide geometry that is already placed, so
             this asks whether growth is better spent refining pieces that
             already work or seeding new ones.

The two families share the default point (0.35/0.15/0.50), which is submitted
once as the ``default`` arm and reported inside both. Nine arms x ten pairs --
the five basic ones (a detailed shape against a simple primitive) and the five
hard ones (two detailed silhouettes competing for the same geometry) -- x one
seed = 90 jobs, all under ``results/opweights/``.

``lambda_count`` is 0 for every job. Piece count is the thing being measured
here, so nothing else is allowed to ration it: the acceptance gate is pure loss
improvement and the piece count that comes out is the operation mix's own doing.

Metrics come from ``run_final.py``, which logs step, mean IoU, piece count and
the cumulative add/split/delete counts at every evaluation in ``history.csv``
(IoU and pieces over steps, and via the sweep over operation weight), and
exports the two final rendered camera views per job. Runs stop when both mean
IoU and loss plateau, never before --min-steps (1000) and never past --steps
(4000) or --max-hours.

Everything else is the fixed configuration shared with the hard, piece-count
and pruning batches: fan-fill triangulation, swept-volume resolution 256, 100%
of SRD additions drawn from the swept volume, the planar overlap test with its
hard repair pass, and a silhouette:negative-space weight ratio of 2 at a fixed
sum of 4.5 (-> 3.0/1.5).

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_opweights.py --dry-run

    # Submit 10 pairs x 9 arms x 1 seed = 90 jobs
    python submit_opweights.py

    # After jobs finish: aggregate into opweights/<name>/collected/
    python submit_opweights.py --sweep-name <name> --collect
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

OPWEIGHTS_ROOT = _PROJECT_ROOT / "results" / "opweights"

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

# The mix every arm is a perturbation of.
DEFAULT_ADD = 0.35
DEFAULT_DELETE = 0.15
DEFAULT_SPLIT = 0.50

# Swept values per family. The default sits at the top of the add sweep (it is
# the upper bound) and in the middle of the split sweep.
ADD_VALUES = (0.05, 0.10, 0.175, 0.25, 0.35)
SPLIT_VALUES = (0.20, 0.35, 0.50, 0.65, 0.80)

BASE_ARM = "srd"

# Piece count is the measured outcome, so no piece-count penalty rations it.
LAMBDA_COUNT = 0.0

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


def _tag(value: float) -> str:
    """0.175 -> 'p175', for job names."""
    return f"p{value:g}".replace("0.", "").replace(".", "")


def _build_arms() -> dict[str, dict]:
    """Arm name -> {add, delete, split, families}.

    The default mix belongs to both families and is submitted once.
    """
    arms: dict[str, dict] = {
        "default": {
            "add": DEFAULT_ADD,
            "delete": DEFAULT_DELETE,
            "split": DEFAULT_SPLIT,
            "families": ("add", "split"),
            "add_x": DEFAULT_ADD,
            "split_x": DEFAULT_SPLIT,
        }
    }

    # --- add family: delete:split held at the default 0.15:0.50 ratio
    delete_share = DEFAULT_DELETE / (DEFAULT_DELETE + DEFAULT_SPLIT)
    for value in ADD_VALUES:
        if abs(value - DEFAULT_ADD) < 1e-9:
            continue
        rest = 1.0 - value
        arms[f"add_{_tag(value)}"] = {
            "add": value,
            "delete": rest * delete_share,
            "split": rest * (1.0 - delete_share),
            "families": ("add",),
            "add_x": value,
            "split_x": rest * (1.0 - delete_share),
        }

    # --- split family: add:delete held at the default 0.35:0.15 (7:3) ratio
    add_share = DEFAULT_ADD / (DEFAULT_ADD + DEFAULT_DELETE)
    for value in SPLIT_VALUES:
        if abs(value - DEFAULT_SPLIT) < 1e-9:
            continue
        rest = 1.0 - value
        arms[f"split_{_tag(value)}"] = {
            "add": rest * add_share,
            "delete": rest * (1.0 - add_share),
            "split": value,
            "families": ("split",),
            "add_x": rest * add_share,
            "split_x": value,
        }

    return arms


ARM_SETTINGS = _build_arms()
ARMS = tuple(ARM_SETTINGS)


def _family_arms(family: str) -> list[str]:
    """Arms belonging to a family, ordered by that family's swept value."""
    members = [name for name, arm in ARM_SETTINGS.items() if family in arm["families"]]
    key = "add_x" if family == "add" else "split_x"
    return sorted(members, key=lambda name: ARM_SETTINGS[name][key])


def _weights(ratio: float, weight_sum: float) -> tuple[float, float]:
    """Split weight_sum into (silhouette, negative_space) at the given ratio."""
    negative_space = weight_sum / (1.0 + ratio)
    return weight_sum - negative_space, negative_space


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the SRD operation-weight sweep to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Sweep directory under results/opweights/ (default: ow_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--arms",
        nargs="*",
        choices=sorted(ARMS),
        default=sorted(ARMS),
        help="Operation-mix arms to run.",
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
    parser.add_argument("--srd-candidates", type=int, default=32)
    parser.add_argument("--lambda-count", type=float, default=LAMBDA_COUNT)
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
        for arm in args.arms:
            settings = ARM_SETTINGS[arm]
            job_name = f"ow_{pair}_{arm}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", BASE_ARM,
                "--add-weight", f"{settings['add']:.6g}",
                "--delete-weight", f"{settings['delete']:.6g}",
                "--split-weight", f"{settings['split']:.6g}",
                "--srd-candidates", str(args.srd_candidates),
                "--lambda-count", f"{args.lambda_count:g}",
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
                "arm": arm,
                "add": settings["add"],
                "delete": settings["delete"],
                "split": settings["split"],
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
    lines = ["job_id\tjob_name\tpair\tarm\tadd_weight\tdelete_weight\tsplit_weight\toutput_dir"]
    for row in rows:
        lines.append(
            f"{row.get('job_id', '-')}\t{row['name']}\t{row['pair']}\t{row['arm']}\t"
            f"{row['add']:.6g}\t{row['delete']:.6g}\t{row['split']:.6g}\t{row['output_dir']}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

_SUMMARY_KEYS = (
    "add_weight", "delete_weight", "split_weight", "srd_candidates",
    "lambda_count", "seed", "overlap_mode", "stop_reason",
    "final_step", "final_loss",
    "final_mean_iou", "final_view1_iou", "final_view2_iou",
    "best_mean_iou", "best_mean_iou_step",
    "final_mean_spill", "final_mean_precision", "final_mean_coverage",
    "final_patches", "start_patches", "max_patches", "min_patches", "mean_patches",
    "final_overlap",
    # The operation counts this sweep exists to compare.
    "srd_total_adds", "srd_total_splits", "srd_total_growth", "srd_total_deletes",
    "total_seconds",
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
    """Split 'ow_<pair>_<arm>' back into its parts."""
    stripped = re.sub(r"^ow_", "", job_name)
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
        rows.append(row)

    group_order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    rows.sort(key=lambda r: (group_order.get(r["pair"], 99), arm_order.get(r["arm"], 99)))

    # --- per-job summary
    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- every metric over steps, every job: IoU, piece count and the
    #     cumulative add/split/delete counts, tagged by arm and its weights
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
        settings = ARM_SETTINGS.get(row["arm"], {})
        prefix = (
            f"{row['pair']},{row['arm']},{settings.get('add', 0):.6g},"
            f"{settings.get('delete', 0):.6g},{settings.get('split', 0):.6g}"
        )
        if not curve_lines:
            curve_lines.append(
                f"pair,arm,add_weight,delete_weight,split_weight,{body[0]}"
            )
        for line in body[1:]:
            if line.strip():
                curve_lines.append(f"{prefix},{line}")
    curves_path.write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

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

    # --- headline: the sweep curve itself, one table per family. The default
    #     arm appears in both, since both families pass through it.
    mean_columns = [
        "final_mean_iou", "best_mean_iou", "final_patches", "max_patches",
        "srd_total_adds", "srd_total_splits", "srd_total_deletes",
        "final_loss", "final_mean_spill", "final_mean_coverage",
        "final_step", "total_seconds",
    ]
    by_arm_path = collected / "by_arm.tsv"
    lines = [
        "family\tswept_value\tgroup\tarm\tn\tadd_weight\tdelete_weight\tsplit_weight\t"
        + "\t".join(mean_columns)
    ]
    for family in ("add", "split"):
        key = "add_x" if family == "add" else "split_x"
        for label, members_of in (
            ("basic", set(BASIC_PAIRS)),
            ("hard", set(HARD_PAIRS)),
            ("all", set(ALL_PAIRS)),
        ):
            for arm in _family_arms(family):
                members = [r for r in rows if r["arm"] == arm and r["pair"] in members_of]
                if not members:
                    continue
                settings = ARM_SETTINGS[arm]
                values = [_mean([m[column] for m in members]) for column in mean_columns]
                lines.append(
                    f"{family}\t{settings[key]:.6g}\t{label}\t{arm}\t{len(members)}\t"
                    f"{settings['add']:.6g}\t{settings['delete']:.6g}\t"
                    f"{settings['split']:.6g}\t" + "\t".join(values)
                )
    by_arm_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- console view
    print(f"{'job':<40} {'pair':<16} {'arm':<12} {'f_pc':>5} {'max_pc':>7} "
          f"{'f_iou':>8} {'adds':>6} {'splits':>7} {'dels':>6} {'step':>6} {'stop':<10}")
    print("-" * 132)
    for row in rows:
        print(
            f"{row['job']:<40} {row['pair']:<16} {row['arm']:<12} "
            f"{row['final_patches']:>5} {row['max_patches']:>7} "
            f"{row['final_mean_iou']:>8} {row['srd_total_adds']:>6} "
            f"{row['srd_total_splits']:>7} {row['srd_total_deletes']:>6} "
            f"{row['final_step']:>6} {row['stop_reason']:<10}"
        )

    print("\nBy arm -- IoU and pieces against operation weight:")
    print(f"{'family':<7} {'value':>6} {'group':<7} {'arm':<12} {'n':>3} "
          f"{'f_iou':>8} {'f_pc':>6} {'max_pc':>7} {'adds':>7} {'splits':>7} "
          f"{'dels':>7} {'step':>7}")
    for line in lines[1:]:
        fields = line.split("\t")
        family, value, group, arm, n = fields[0], fields[1], fields[2], fields[3], fields[4]
        values = dict(zip(mean_columns, fields[8:]))
        print(
            f"{family:<7} {value:>6} {group:<7} {arm:<12} {n:>3} "
            f"{values['final_mean_iou']:>8} {values['final_patches']:>6} "
            f"{values['max_patches']:>7} {values['srd_total_adds']:>7} "
            f"{values['srd_total_splits']:>7} {values['srd_total_deletes']:>7} "
            f"{values['final_step']:>7}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv and are "
              f"absent from curves.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path, by_arm_path):
        print(f"  {path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    sweep_name = args.sweep_name or f"ow_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = OPWEIGHTS_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    print(
        f"[Sweep] opweights/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(args.arms)} arm(s) x 1 seed)"
    )
    for family in ("add", "split"):
        print(f"[Sweep]   {family} family:")
        for arm in _family_arms(family):
            if arm not in args.arms:
                continue
            settings = ARM_SETTINGS[arm]
            marker = "  <- default" if arm == "default" else ""
            print(
                f"[Sweep]     {arm:<12} add={settings['add']:.4f} "
                f"delete={settings['delete']:.4f} split={settings['split']:.4f}{marker}"
            )
    print(
        f"[Sweep] base arm={BASE_ARM}, lambda_count={args.lambda_count:g}, "
        f"weights: silhouette={silhouette:g}, "
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
        print("Monitor with: squeue --me | grep ow_")
        print(f"Aggregate when done: python submit_opweights.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
