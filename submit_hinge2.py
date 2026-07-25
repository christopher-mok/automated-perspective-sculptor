"""Submit the repaired-hinge piece-count batch to SLURM.

The countobj batch (co_20260722_1223) established that scoring rewrites on

    J = n_patches + count_lambda * hinge(target - mean_iou)^2

is the right frame -- it reached 0.90 IoU with 2-3x fewer pieces than any
loss-scored arm -- but eight of its ten runs finished just *under* the target
(0.885, 0.889, 0.897, 0.897, 0.906 ...), which is the objective doing exactly
what it was told. At count_power=2 the marginal price of a piece is
2*lambda*deficit, which vanishes at the threshold: standing exactly on target,
a deletion costing up to sqrt(1/count_lambda) = 0.0224 IoU is free. The runs
were not failing to converge, they were converging to a band below the target.

This batch changes five things, which are one mechanism rather than five:

1.  count_power=1. The linear hinge is the actual Lagrangian of "minimise n
    subject to IoU >= target", with constant slope lambda up to the kink. It
    also makes lambda interpretable: a deletion is accepted exactly when it
    costs less than 1/lambda of IoU, so the reciprocal *is* the tolerance.
    Verified against the implementation: at lambda=500 the largest free
    deletion is 0.0019 IoU, against 0.0223 under the old power=2/lambda=2000.

2.  --count-hard-floor. The hinge prices infeasibility; nothing refuses it.
    Once the run has cleared the target, a rewrite landing back below it is
    rejected outright instead of bought at rate lambda. This is what turns
    "0.9-ish" into ">= target".

3.  --count-start-mode on_target. The old runs shed capacity while the fit was
    still climbing -- co_horse_circle went 20 -> 13 pieces before ever touching
    0.90 -- which judges pieces on unconverged evidence and caps the reachable
    IoU. Here the count objective stays off until IoU has held at
    target + margin, so shedding starts from a converged, feasible sculpture.
    The no-count-pressure prune arm reached 0.944/0.964 on these same pairs,
    so that ceiling is real and worth protecting.

4.  --shed-patience-evals. Once shedding starts, IoU plateaus by design, so the
    ordinary IoU/loss plateau rule trips immediately: every countobj run
    stopped at step 1000-1250 of 4000, and co_cat_face_bass was still improving
    (0.894 -> 0.919 at 7 pieces) when it was killed. Convergence during the
    shed phase means "no accepted deletion for N evaluations", and never fires
    below target.

5.  --delete-eval-steps. A deletion costs ~0.02 IoU on impact and recovers over
    ~100+ steps (co_water_fire: 0.906 -> 0.884 -> 0.905), but deletions were
    scored after a single refit step -- at the bottom of the dip. Merely a
    conservative bias before, fatal alongside (2), which would veto every
    useful deletion at the instant it happens.

Plus --count-margin-reward, a gentle downward tilt on the hinge's flat half so
margin is worth something and a run does not drift onto the threshold where
noise and the post-deletion dip flip it infeasible. It is a tiebreak: at 5, a
piece must buy 0.2 IoU to justify itself, far above the ~0.005 one actually
buys near convergence, so it never changes a piece-count decision.

The target is 0.91 rather than 0.90, so that the residual dip lands on 0.90
rather than under it, and the shed phase opens at 0.93.

Stage 1 (--stage lambda) is the lambda sweep at the full stack: three arms x
seven pairs = 21 jobs. Since lambda is now a tolerance, the three arms are
"a piece may cost 0.005 / 0.002 / 0.0005 IoU". Stage 2 (--stage ablate) holds
lambda at 500 and removes one component at a time, to attribute whatever
stage 1 shows. The countobj batch is the baseline for both -- it already
covers all seven pairs under the old configuration, so no baseline arm is
re-run here.

Two of the seven pairs (acm_scf, teapot_droplets) never reached 0.90 in any arm
of any batch, at any piece count, including unpenalised 24-27 piece runs. They
are here as the infeasibility control: for them the fit phase should never
clear target + margin, the run should stay loss-scored throughout, and
shed_entered_step should come back -1.

Examples
--------
    # Preview what would be submitted (writes scripts, submits nothing)
    python submit_hinge2.py --dry-run

    # Stage 1: the lambda sweep, 3 arms x 7 pairs = 21 jobs
    python submit_hinge2.py

    # Stage 2: component ablations at lambda=500, 5 arms x 7 pairs = 35 jobs
    python submit_hinge2.py --stage ablate

    # After jobs finish
    python submit_hinge2.py --sweep-name <name> --collect
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

HINGE2_ROOT = _PROJECT_ROOT / "results" / "hinge2"

# The seven pairs under test. The first five have reached 0.90 in at least one
# arm of an earlier batch; the last two never have, and are the infeasibility
# control -- see the module docstring.
FEASIBLE_PAIRS = (
    "sun_moon",
    "water_fire",
    "cat_face_bass",
    "dance_argument",
    "horse_circle",
)

INFEASIBLE_PAIRS = (
    "acm_scf",
    "teapot_droplets",
)

ALL_PAIRS = FEASIBLE_PAIRS + INFEASIBLE_PAIRS

BASE_ARM = "srd"

# silhouette : negative_space, at a fixed sum, under --area-normalized-view-loss.
#
# The areaweights sweep (aw_20260722_2238, six pairs x five ratios) measured
# this directly, and best mean IoU falls monotonically as the ratio rises:
#
#     ratio    0.25    0.5     1.0     2.0     4.0
#     mean    0.905   0.902   0.867   0.851   0.833
#
# 0.25 wins five of six pairs outright (0.5 takes acm_scf by 0.04) and beats
# the 2.0 this batch previously ran at by 0.038 mean IoU -- larger than the
# gap between any two arms of the h2 batch itself. Weighting negative space
# roughly 4:1 over silhouette buys IoU because IoU charges for spill twice,
# once as a miss and once as union it did not earn.
#
# Two caveats carried from the sweep. 0.25 is the *edge* of the measured grid,
# so the optimum may lie below it and this is a lower bound on the gain, not a
# located optimum. And the ratio was measured under area normalization, where
# both terms read as mean squared alpha error per unit of their own region;
# without it the effective ratio drifts with each target's foreground fraction
# and 0.25 does not mean the same thing, which is why AREA_NORMALIZED_VIEW_LOSS
# below travels with it rather than being an independent switch.
WEIGHT_RATIO = 0.25
WEIGHT_SUM = 4.5

# The h2_20260723_0259 batch ran without this, at ratio 2. Both change here.
AREA_NORMALIZED_VIEW_LOSS = True

# Piece-count penalty during the loss-scored *fit* phase (count_start_mode
# on_target keeps the count objective off until target is held, so the fit is
# scored on loss + lambda_count * n_patches). run_final.py defaults this to
# 0.05, which charges every add 0.05 in loss units and stalls growth: the
# h2_20260723_0259 batch started at 20 pieces and added exactly zero, so the
# fit never grew into the pieces the harder pairs need to clear target and the
# hinge had nothing feasible to shed. Set it to 0 so the fit grows unrationed
# to the IoU ceiling -- the same lambda_count=0 the areaweights arm used to
# reach 27-31 pieces at 0.90-0.96 -- and let the hinge do all the rationing
# afterwards, in the shed phase, where it belongs. This is the "protect the
# prune ceiling" point of the module docstring made literal.
FIT_LAMBDA_COUNT = 0.0

# Every arm inherits these; an arm's own entry overrides on conflict. This is
# the full repaired stack, and stage 1 varies only count_lambda within it.
FULL_STACK = {
    "count_power": "1",
    "count_margin_reward": "5",
    "count_hard_floor": True,
    "count_start_mode": "on_target",
    "count_start_margin": "0.02",
    "count_start_patience": "2",
    "delete_eval_steps": "15",
    "shed_patience_evals": "8",
}

# Stage 1: lambda is now a tolerance, so these read as "a piece may cost at
# most this much IoU": 1/200 = 0.005, 1/500 = 0.002, 1/2000 = 0.0005.
LAMBDA_ARMS = {
    "l200": {"count_lambda": "200"},
    "l500": {"count_lambda": "500"},
    "l2000": {"count_lambda": "2000"},
}

# Stage 2: hold lambda at 500 and drop one component each, so any effect
# stage 1 shows can be attributed. "p2old" restores the old hinge shape inside
# the new scaffolding, separating "the shape was wrong" from "the schedule was
# wrong" -- the single most important control in the batch.
ABLATION_ARMS = {
    "noshed": {"count_lambda": "500", "count_start_mode": "immediate"},
    "nofloor": {"count_lambda": "500", "count_hard_floor": False},
    "fastdel": {"count_lambda": "500", "delete_eval_steps": "1"},
    "nomargin": {"count_lambda": "500", "count_margin_reward": "0"},
    "p2old": {"count_lambda": "2000", "count_power": "2", "count_margin_reward": "0"},
}

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
        description="Submit the repaired-hinge piece-count batch to SLURM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sweep-name",
        default=None,
        help="Batch directory under results/hinge2/ (default: h2_<timestamp>).",
    )
    parser.add_argument("--pairs", nargs="*", default=None, help="Override the pair list.")
    parser.add_argument(
        "--stage",
        choices=("lambda", "ablate", "both"),
        default="lambda",
        help="'lambda' sweeps count_lambda at the full stack; 'ablate' holds "
             "lambda at 500 and removes one component per arm.",
    )
    parser.add_argument("--arms", nargs="*", default=None, help="Override the arm list.")
    parser.add_argument("--seed", type=int, default=0, help="Single trial seed per pair.")

    objective = parser.add_argument_group("piece-count objective")
    objective.add_argument(
        "--count-target-iou",
        type=float,
        default=0.91,
        help="0.91 rather than 0.90 so the residual dip lands on 0.90.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
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


def _arm_table(stage: str) -> dict[str, dict]:
    if stage == "lambda":
        return dict(LAMBDA_ARMS)
    if stage == "ablate":
        return dict(ABLATION_ARMS)
    return {**LAMBDA_ARMS, **ABLATION_ARMS}


def _arm_flags(overrides: dict) -> list[str]:
    """Render FULL_STACK + arm overrides as run_final.py flags.

    Booleans are store_true switches: present when True, absent when False,
    which is how an arm drops a component.
    """
    settings = {**FULL_STACK, **overrides}
    flags: list[str] = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                flags.append(flag)
        else:
            flags.extend([flag, str(value)])
    return flags


def _build_jobs(args: argparse.Namespace, sweep_dir: Path, arms: dict) -> list[dict]:
    image_dir = _PROJECT_ROOT / "images"
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)

    jobs: list[dict] = []
    for pair in args.pairs:
        target1, target2 = IMAGE_PAIRS[pair]
        for arm_name, overrides in arms.items():
            job_name = f"h2_{pair}_{arm_name}"
            output_dir = sweep_dir / job_name
            command = [
                args.python, "run_final.py",
                "--target1", str(image_dir / target1),
                "--target2", str(image_dir / target2),
                "--arm", BASE_ARM,
                "--count-objective",
                "--count-target-iou", f"{args.count_target_iou:g}",
                "--lambda-count", f"{FIT_LAMBDA_COUNT:g}",
                *_arm_flags(overrides),
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
                *(["--area-normalized-view-loss"] if AREA_NORMALIZED_VIEW_LOSS else []),
                "--render-scale", str(args.render_scale),
                "--max-hours", str(args.max_hours),
                "--device", args.device,
                "--output-dir", str(output_dir),
            ]
            jobs.append({
                "name": job_name,
                "pair": pair,
                "arm": arm_name,
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
    "count_target_iou", "count_lambda", "count_power", "count_margin_reward",
    "count_hard_floor", "count_start_mode", "delete_eval_steps",
    "shed_patience_evals", "seed", "stop_reason", "final_step",
    # Did the constraint hold, and what did it cost in pieces?
    "count_target_reached", "count_final_deficit",
    "final_mean_iou", "best_mean_iou", "final_patches",
    # The fit-then-shed record: where the shed phase opened and what it shed.
    "shed_entered_step", "shed_entry_patches", "shed_entry_mean_iou",
    "shed_evals", "min_iou_after_target",
    "patches_at_threshold", "patches_shed_after_threshold",
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
        name = re.sub(r"^h2_", "", report.parent.name)
        arm = name.rsplit("_", 1)[1]
        pair = name.rsplit("_", 1)[0]
        row = {"job": report.parent.name, "pair": pair, "arm": arm}
        for key in _SUMMARY_KEYS:
            row[key] = _grab(text, key)
        rows.append(row)

    order = {pair: index for index, pair in enumerate(ALL_PAIRS)}
    rows.sort(key=lambda r: (order.get(r["pair"], 99), r["arm"]))

    summary_path = collected / "summary.tsv"
    columns = ["job", "pair", "arm"] + list(_SUMMARY_KEYS)
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(row.get(column, "-") for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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

    # The headline table. "held" is the whole point of the batch: whether the
    # run finished at or above target, which the countobj batch managed twice
    # in ten. "min_after" is the floor's audit trail -- it should never sit
    # below the target once target_reached has latched.
    print(f"{'pair':<17} {'arm':<9} {'held':>5} {'f_iou':>7} {'min_after':>9} "
          f"{'pcs':>4} {'shed@':>6} {'entry_pc':>8} {'dels':>5} {'step':>5} {'stop':<14}")
    print("-" * 104)
    for row in rows:
        print(
            f"{row['pair']:<17} {row['arm']:<9} {row['count_target_reached']:>5} "
            f"{row['final_mean_iou']:>7} {row['min_iou_after_target']:>9} "
            f"{row['final_patches']:>4} {row['shed_entered_step']:>6} "
            f"{row['shed_entry_patches']:>8} {row['srd_total_deletes']:>5} "
            f"{row['final_step']:>5} {row['stop_reason']:<14}"
        )

    if missing_curves:
        print(f"\n[warn] {missing_curves} job(s) had no history.csv")

    print(f"\nWritten to {collected}:")
    for path in (summary_path, curves_path):
        print(f"  {path.name}")
    print(f"  views/ ({view_count} png)")


def main() -> None:
    args = _parse_args()

    if args.pairs is None:
        args.pairs = list(ALL_PAIRS)
    unknown = [pair for pair in args.pairs if pair not in IMAGE_PAIRS]
    if unknown:
        raise SystemExit(f"Unknown pair(s): {', '.join(unknown)}")

    arms = _arm_table(args.stage)
    if args.arms is not None:
        unknown_arms = [a for a in args.arms if a not in arms]
        if unknown_arms:
            raise SystemExit(f"Unknown arm(s): {', '.join(unknown_arms)}")
        arms = {name: arms[name] for name in args.arms}

    sweep_name = args.sweep_name or f"h2_{datetime.now().strftime('%Y%m%d_%H%M')}"
    sweep_dir = HINGE2_ROOT / sweep_name

    if args.collect:
        _collect(sweep_dir)
        return

    sweep_dir.mkdir(parents=True, exist_ok=True)
    jobs = _build_jobs(args, sweep_dir, arms)
    silhouette, negative_space = _weights(args.weight_ratio, args.weight_sum)
    print(
        f"[Batch] hinge2/{sweep_name}: {len(jobs)} job(s) "
        f"({len(args.pairs)} pair(s) x {len(arms)} arm(s) x 1 seed), stage={args.stage}"
    )
    print(
        f"[Batch]   J = n_patches + lambda * max(0, "
        f"{args.count_target_iou:g} - mean_iou)^p - mu * max(0, mean_iou - "
        f"{args.count_target_iou:g})"
    )
    for arm_name, overrides in arms.items():
        settings = {**FULL_STACK, **overrides}
        lam = float(settings["count_lambda"])
        power = float(settings["count_power"])
        tolerance = (
            f"{1.0 / lam:.4f}" if power == 1.0 else f"{(1.0 / lam) ** 0.5:.4f}"
        )
        print(
            f"[Batch]   {arm_name:<9} lambda={lam:<6g} p={power:g} "
            f"mu={settings['count_margin_reward']:<3} "
            f"floor={str(settings['count_hard_floor']):<5} "
            f"start={settings['count_start_mode']:<9} "
            f"del_steps={settings['delete_eval_steps']:<3} "
            f"-> free deletion <= {tolerance} IoU"
        )
    print(
        f"[Batch] base arm={BASE_ARM}, weights: silhouette={silhouette:g}, "
        f"negative_space={negative_space:g} (ratio {args.weight_ratio:g}, "
        f"area_normalized={AREA_NORMALIZED_VIEW_LOSS}), "
        f"steps {args.min_steps}..{args.steps} with early stop"
    )
    print(
        f"[Batch] infeasibility control: {', '.join(INFEASIBLE_PAIRS)} "
        f"(expect shed_entered_step=-1)"
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
        print("Monitor with: squeue --me | grep h2_")
        print(f"Aggregate when done: python submit_hinge2.py "
              f"--sweep-name {sweep_name} --collect")


if __name__ == "__main__":
    main()
