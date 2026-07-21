"""Single-trial runner for the final comparison, ablation and overlap batches.

One invocation is one trial (one arm, one seed), so trials spread across SLURM
jobs. Every run records loss and IoU on a fixed step interval, not just at the
end, because the questions these batches answer are as much about *when* a run
gets somewhere as about where it ends up -- the swept-volume ablation in
particular is expected to reach a similar final loss more slowly.

Arms
----
    basic          SRD disabled entirely (the baseline method).
    srd            The full method: SRD with swept-volume-guided additions.
    srd_no_swept   SRD with swept-volume-guided additions disabled; every
                   other SRD component (rule-based deletion, splitting) stays
                   on. This is the only ablation the final batch runs.

Fan-fill triangulation is on by default so geometry is a property of the run
rather than of whichever triangulation library the environment happens to
have. Swept-volume resolution defaults to 256 and every SRD addition is drawn
from the swept volume.

Outputs into ``--output-dir``:
    report.txt    human-readable report, parsed by submit_final.py
    history.csv   step,seconds,loss,mean_iou,view1_iou,view2_iou,patches,
                  overlap,overlap_repaired_pairs

Example:
    python run_final.py --target1 images/horse.png --target2 images/circle.png \
        --arm srd --seed 0 --steps 2000 --output-dir results/final/run
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from run_ablation import (
    _HANGING_PLANE_Y,
    _load_target_image_with_border,
    _make_scene_cameras,
)

_PROJECT_ROOT = Path(__file__).resolve().parent

ARMS = ("basic", "srd", "srd_no_swept")

# Absolute mean-IoU milestones whose first crossing is reported, so a run that
# converges faster shows up as smaller steps/seconds at the same milestone
# rather than only as a different endpoint.
IOU_MILESTONES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)

# Fractions of a run's *own* best IoU, which stay meaningful when a pair never
# reaches the absolute milestones at all.
RELATIVE_MILESTONES = (0.90, 0.95, 0.99)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one final-batch trial, tracking loss and IoU over time.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", required=True, help="View 2 target image path.")
    parser.add_argument(
        "--arm",
        choices=ARMS,
        required=True,
        help="'basic' = no SRD, 'srd' = full method, "
             "'srd_no_swept' = SRD without swept-volume-guided additions.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Trial seed.")
    parser.add_argument("--steps", type=int, default=2000, help="Optimization steps.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Sample loss and IoU every N steps (one extra no-grad render).",
    )
    parser.add_argument(
        "--no-fan-fill",
        action="store_true",
        help="Control arm: keep the default earcut triangulation.",
    )

    run = parser.add_argument_group("run settings")
    run.add_argument("--n-patches", type=int, default=20)
    run.add_argument("--swept-resolution", type=int, default=256)
    run.add_argument(
        "--swept-spawn-fraction",
        type=float,
        default=1.0,
        help="Fraction of SRD additions drawn from the swept volume.",
    )
    run.add_argument("--lr", type=float, default=3.5e-3)
    run.add_argument("--device", default="cuda")
    run.add_argument("--palette", default="")
    run.add_argument("--hanging-plane-size", type=float, default=5.0)
    run.add_argument("--srd-interval", type=int, default=50)
    run.add_argument("--srd-candidates", type=int, default=32)
    run.add_argument("--max-hours", type=float, default=10.0, help="Hard wall-clock cap.")
    run.add_argument("--output-dir", required=True, help="Report/CSV directory.")

    weights = parser.add_argument_group("view-loss weights")
    weights.add_argument(
        "--silhouette-weight",
        type=float,
        default=3.6,
        help="Silhouette term weight, applied to both views.",
    )
    weights.add_argument(
        "--negative-space-weight",
        type=float,
        default=0.9,
        help="Negative-space term weight, applied to both views. The default "
             "pair is a silhouette:negative-space ratio of 4 at a fixed sum "
             "of 4.5.",
    )

    overlap = parser.add_argument_group("overlap test")
    overlap.add_argument(
        "--overlap-mode",
        choices=("sphere", "planar"),
        default="planar",
        help="'sphere' is the legacy orientation-blind bounding-sphere test; "
             "'planar' clips both outlines against the plane intersection line.",
    )
    overlap.add_argument(
        "--overlap-repair",
        dest="overlap_repair",
        action="store_true",
        default=True,
        help="Hard-separate pairs the exact planar test finds interpenetrating.",
    )
    overlap.add_argument(
        "--no-overlap-repair",
        dest="overlap_repair",
        action="store_false",
        help="Soft penalty only; no hard repair pass.",
    )
    overlap.add_argument("--overlap-repair-interval", type=int, default=5)
    return parser.parse_args()


def _srd_config(args: argparse.Namespace) -> dict[str, object] | None:
    if args.arm == "basic":
        return None
    return {
        "enabled": True,
        "interval": args.srd_interval,
        "candidate_count": args.srd_candidates,
        "disable_swept_volume_adds": args.arm == "srd_no_swept",
        "swept_volume_spawn_fraction": args.swept_spawn_fraction,
        "loss_only_deletion": False,
        "disable_splitting": False,
    }


def _write_history_csv(path: Path, history: list[dict]) -> None:
    columns = [
        "step", "seconds", "loss", "mean_iou", "view1_iou", "view2_iou",
        "patches", "overlap", "overlap_repaired_pairs",
    ]
    lines = [",".join(columns)]
    for entry in history:
        lines.append(
            f"{entry['step']},{entry['seconds']:.3f},{entry['loss']:.6f},"
            f"{entry['mean_iou']:.6f},{entry['view1_iou']:.6f},"
            f"{entry['view2_iou']:.6f},{entry['patches']},"
            f"{entry['overlap']:.8f},{entry['overlap_repaired_pairs']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _first_crossing(history: list[dict], threshold: float) -> dict | None:
    """First evaluation whose mean_iou reached ``threshold``."""
    for entry in history:
        if entry["mean_iou"] >= threshold:
            return entry
    return None


def _milestone_lines(history: list[dict]) -> list[str]:
    """Steps and seconds to each IoU milestone -- the convergence-speed record."""
    lines: list[str] = []
    for threshold in IOU_MILESTONES:
        crossing = _first_crossing(history, threshold)
        tag = f"{threshold:.2f}".replace(".", "p")
        if crossing is None:
            lines.append(f"  iou_{tag}_steps=-1")
            lines.append(f"  iou_{tag}_seconds=-1")
        else:
            lines.append(f"  iou_{tag}_steps={crossing['step']}")
            lines.append(f"  iou_{tag}_seconds={crossing['seconds']:.3f}")

    best = max((entry["mean_iou"] for entry in history), default=0.0)
    for fraction in RELATIVE_MILESTONES:
        crossing = _first_crossing(history, best * fraction)
        tag = f"{int(round(fraction * 100))}"
        if crossing is None:
            lines.append(f"  rel_{tag}pct_steps=-1")
            lines.append(f"  rel_{tag}pct_seconds=-1")
        else:
            lines.append(f"  rel_{tag}pct_steps={crossing['step']}")
            lines.append(f"  rel_{tag}pct_seconds={crossing['seconds']:.3f}")
    return lines


def _write_report(
    path: Path,
    args: argparse.Namespace,
    started_at: str,
    history: list[dict],
    stop_reason: str,
    total_seconds: float,
    setup_seconds: float,
    srd_stats: object | None,
) -> None:
    best = max(history, key=lambda entry: entry["mean_iou"]) if history else None
    best_loss = min(history, key=lambda entry: entry["loss"]) if history else None
    final = history[-1] if history else None

    ratio = (
        args.silhouette_weight / args.negative_space_weight
        if args.negative_space_weight else float("inf")
    )
    lines = [
        "Final-batch run report",
        f"started={started_at}",
        f"arm={args.arm}",
        f"seed={args.seed}",
        f"geometry={'fan_fill' if not args.no_fan_fill else 'earcut'}",
        f"target1={args.target1}",
        f"target2={args.target2}",
        f"steps={args.steps}",
        f"eval_interval={args.eval_interval}",
        f"n_patches={args.n_patches}",
        f"swept_volume_resolution={args.swept_resolution}",
        f"swept_volume_spawn_fraction={args.swept_spawn_fraction:.6g}",
        f"srd_enabled={args.arm != 'basic'}",
        f"disable_swept_volume_adds={args.arm == 'srd_no_swept'}",
        f"silhouette_weight={args.silhouette_weight:.6g}",
        f"negative_space_weight={args.negative_space_weight:.6g}",
        f"weight_ratio={ratio:.6g}",
        f"overlap_mode={args.overlap_mode}",
        f"overlap_repair={args.overlap_repair}",
        f"overlap_repair_interval={args.overlap_repair_interval}",
        f"learning_rate={args.lr:.6g}",
        f"device={args.device}",
        "",
        "Result:",
        f"  stop_reason={stop_reason}",
        f"  setup_seconds={setup_seconds:.3f}",
        f"  total_seconds={total_seconds:.3f}",
    ]

    if best is not None and best_loss is not None and final is not None:
        lines.extend([
            f"  final_step={final['step']}",
            f"  final_loss={final['loss']:.6f}",
            f"  final_mean_iou={final['mean_iou']:.6f}",
            f"  final_view1_iou={final['view1_iou']:.6f}",
            f"  final_view2_iou={final['view2_iou']:.6f}",
            f"  final_patches={final['patches']}",
            f"  final_overlap={final['overlap']:.8f}",
            f"  best_mean_iou={best['mean_iou']:.6f}",
            f"  best_mean_iou_step={best['step']}",
            f"  best_loss={best_loss['loss']:.6f}",
            f"  best_loss_step={best_loss['step']}",
        ])

    if srd_stats is not None:
        lines.extend([
            f"  srd_total_adds={getattr(srd_stats, 'total_added', 0)}",
            f"  srd_total_deletes={getattr(srd_stats, 'total_deleted', 0)}",
        ])

    lines.append("")
    lines.append("Convergence:")
    lines.extend(_milestone_lines(history))
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    started = datetime.now()
    started_at = started.isoformat(timespec="seconds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from core.initialization import initialize_patches
    from core.optimizer import SceneOptimizer
    from core.patch import set_fan_fill
    from core.swept_volume import SweptVolume

    # Before any patch is constructed, so every mesh built in this process uses
    # the same triangulation.
    set_fan_fill(not args.no_fan_fill)

    setup_started = time.perf_counter()

    cameras = _make_scene_cameras()
    target1 = _load_target_image_with_border(args.target1)
    target2 = _load_target_image_with_border(args.target2)

    print(f"[Swept volume] building (resolution={args.swept_resolution})...", flush=True)
    swept_volume = SweptVolume.from_images(
        target1,
        target2,
        cameras,
        hanging_plane_size=args.hanging_plane_size,
        resolution=args.swept_resolution,
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    patches = initialize_patches(
        mode="Experimental",
        n_patches=args.n_patches,
        reference_image=target1,
        cameras=cameras,
        device=args.device,
        seed=args.seed,
        swept_volume=swept_volume,
    )
    optimizer = SceneOptimizer(
        patches,
        cameras[0],
        cameras[1],
        target1,
        target2,
        palette=args.palette,
        lr=args.lr,
        device=args.device,
        hanging_plane_size=args.hanging_plane_size,
        hanging_plane_y=_HANGING_PLANE_Y,
        srd_config=_srd_config(args),
        swept_volume=swept_volume,
        silhouette_weight=args.silhouette_weight,
        negative_space_weight=args.negative_space_weight,
        overlap_mode=args.overlap_mode,
        overlap_repair=args.overlap_repair,
        overlap_repair_interval=args.overlap_repair_interval,
    )

    setup_seconds = time.perf_counter() - setup_started
    print(
        f"[Run] arm={args.arm}, seed={args.seed}, steps={args.steps}, "
        f"overlap={args.overlap_mode} (repair={args.overlap_repair}), "
        f"weights={args.silhouette_weight:g}/{args.negative_space_weight:g}, "
        f"setup={setup_seconds:.1f}s",
        flush=True,
    )

    max_seconds = args.max_hours * 3600.0
    optimization_started = time.perf_counter()

    history: list[dict] = []
    stop_reason = "steps"

    for step_index in range(1, args.steps + 1):
        step_metrics = optimizer.step(step_index, args.steps)

        is_eval = (
            step_index % args.eval_interval == 0
            or step_index == 1
            or step_index == args.steps
        )
        if not is_eval:
            continue

        metrics, _, _ = optimizer.evaluate_snapshot()
        elapsed = time.perf_counter() - optimization_started
        entry = {
            "step": step_index,
            "seconds": elapsed,
            "loss": float(metrics["loss"]),
            "mean_iou": float(metrics.get("mean_iou", 0.0)),
            "view1_iou": float(metrics.get("view1_iou", 0.0)),
            "view2_iou": float(metrics.get("view2_iou", 0.0)),
            "patches": int(metrics["patches"]),
            "overlap": float(metrics.get("overlap", 0.0)),
            "overlap_repaired_pairs": int(step_metrics.get("overlap_repaired_pairs", 0)),
        }
        history.append(entry)

        # Refresh the CSV as we go so a job killed by the walltime still leaves
        # a usable curve behind.
        _write_history_csv(output_dir / "history.csv", history)

        if step_index % (args.eval_interval * 4) == 0 or step_index == 1:
            print(
                f"  step {step_index}/{args.steps}: "
                f"loss={entry['loss']:.6f}, mean_iou={entry['mean_iou']:.6f}, "
                f"patches={entry['patches']}, overlap={entry['overlap']:.3e}, "
                f"{entry['seconds']:.1f}s",
                flush=True,
            )

        if elapsed >= max_seconds:
            stop_reason = "max_hours"
            print(f"  [stop] wall-clock cap reached at step {step_index}", flush=True)
            break

    total_seconds = time.perf_counter() - optimization_started

    _write_history_csv(output_dir / "history.csv", history)
    _write_report(
        output_dir / "report.txt",
        args,
        started_at,
        history,
        stop_reason,
        total_seconds,
        setup_seconds,
        optimizer.srd.stats if optimizer.srd is not None else None,
    )
    print(f"\n[Final] report written to {output_dir / 'report.txt'}", flush=True)


if __name__ == "__main__":
    main()
