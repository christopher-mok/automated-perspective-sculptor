"""Measure time-to-convergence for SRD vs. the base approach on one image pair.

Each invocation runs a single trial (one method, one seed) so that trials can be
spread across SLURM jobs. Convergence is defined two ways and both are reported:

    plateau    loss has stopped improving: the relative gain over a trailing
               window of ``--plateau-window`` steps falls below
               ``--plateau-delta`` for ``--plateau-patience`` consecutive
               evaluations. Loss is used because mean_iou's jitter is a large
               fraction of its range on this project (see _detect_plateau).
    target     mean_iou first reaches ``--target-iou`` (0.85 by default). A run
               may plateau without ever reaching this, in which case the target
               columns are reported as unreached.

The run stops at whichever comes first: plateau, ``--max-steps``, or
``--max-hours``. Reaching the IoU target does *not* stop the run, so the plateau
statistics stay comparable across methods.

Outputs into ``--output-dir``:
    convergence.txt   human-readable report (parsed by submit_convergence.py)
    iou_history.csv   step,seconds,mean_iou,view1_iou,view2_iou,loss,patches

Example:
    python compare_convergence.py --target1 images/horse.png \
        --target2 images/circle.png --method srd --seed 0 \
        --output-dir results/convergence/horse_circle_srd_seed0
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure steps and wall-clock time to convergence for one trial.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", required=True, help="View 2 target image path.")
    parser.add_argument(
        "--method",
        choices=("srd", "base"),
        required=True,
        help="'srd' is the full method; 'base' disables SRD entirely (matches "
             "the 'basic' setting in submit_comparison.py).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Trial seed.")

    convergence = parser.add_argument_group("convergence criteria")
    convergence.add_argument(
        "--eval-interval",
        type=int,
        default=25,
        help="Evaluate mean_iou every N steps (costs one extra no-grad render).",
    )
    convergence.add_argument(
        "--plateau-delta",
        type=float,
        default=0.002,
        help="Converged once loss improves by less than this *fraction* over the window.",
    )
    convergence.add_argument(
        "--plateau-window",
        type=int,
        default=250,
        help="Trailing window (in steps) the plateau gain is measured over.",
    )
    convergence.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help="Consecutive sub-threshold evaluations required to call plateau.",
    )
    convergence.add_argument(
        "--target-iou",
        type=float,
        default=0.85,
        help="Absolute mean_iou milestone to report the crossing of. Reported "
             "only; it never stops the run.",
    )
    convergence.add_argument("--max-steps", type=int, default=5000, help="Hard step cap.")
    convergence.add_argument("--max-hours", type=float, default=10.0, help="Hard wall-clock cap.")

    run = parser.add_argument_group("run settings")
    run.add_argument("--n-patches", type=int, default=20)
    run.add_argument("--swept-resolution", type=int, default=256)
    run.add_argument("--lr", type=float, default=3.5e-3)
    run.add_argument("--device", default="cuda")
    run.add_argument("--palette", default="")
    run.add_argument("--hanging-plane-size", type=float, default=5.0)
    run.add_argument("--srd-interval", type=int, default=50)
    run.add_argument("--srd-candidates", type=int, default=32)
    run.add_argument("--output-dir", required=True, help="Report/CSV directory.")

    weights = parser.add_argument_group("view-loss weights")
    weights.add_argument(
        "--silhouette-weight",
        type=float,
        default=2.0,
        help="Silhouette term weight, applied to both views. SceneOptimizer default.",
    )
    weights.add_argument(
        "--negative-space-weight",
        type=float,
        default=2.5,
        help="Negative-space term weight, applied to both views. SceneOptimizer default.",
    )
    return parser.parse_args()


def _detect_plateau(
    history: list[dict],
    window: int,
    delta: float,
) -> bool:
    """True when loss improved by less than a `delta` fraction over the trailing `window` steps.

    Plateau keys off loss rather than mean_iou: across this project's runs IoU
    tops out near 0.01 after 1000 steps while jittering ~5e-4 step to step, so
    the noise is a large fraction of the signal and an IoU threshold either
    fires immediately or gets reset at random. Loss falls smoothly and
    monotonically over the same steps. IoU is still evaluated and reported.

    The baseline is the best (lowest) loss at or before the window's start
    rather than that sample's raw value, so a transient spike inside the window
    cannot manufacture an artificial gain and reset the plateau counter.
    """
    if not history:
        return False
    current = history[-1]
    window_start = current["step"] - window
    if window_start < 0:
        return False
    prior = [entry for entry in history if entry["step"] <= window_start]
    if not prior:
        return False
    baseline = min(entry["loss"] for entry in prior)
    if baseline <= 0:
        return False
    return ((baseline - current["loss"]) / baseline) < delta


def _write_history_csv(path: Path, history: list[dict]) -> None:
    lines = ["step,seconds,mean_iou,view1_iou,view2_iou,loss,patches"]
    for entry in history:
        lines.append(
            f"{entry['step']},{entry['seconds']:.3f},{entry['mean_iou']:.6f},"
            f"{entry['view1_iou']:.6f},{entry['view2_iou']:.6f},"
            f"{entry['loss']:.6f},{entry['patches']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    args: argparse.Namespace,
    started_at: str,
    history: list[dict],
    plateau: dict | None,
    target: dict | None,
    stop_reason: str,
    total_seconds: float,
    setup_seconds: float,
) -> None:
    best = max(history, key=lambda entry: entry["mean_iou"]) if history else None
    final = history[-1] if history else None

    lines = [
        "Convergence report",
        f"started={started_at}",
        f"method={args.method}",
        f"seed={args.seed}",
        f"target1={args.target1}",
        f"target2={args.target2}",
        f"n_patches={args.n_patches}",
        f"swept_volume_resolution={args.swept_resolution}",
        f"learning_rate={args.lr:.6g}",
        f"device={args.device}",
        f"silhouette_weight={args.silhouette_weight:.6g}",
        f"negative_space_weight={args.negative_space_weight:.6g}",
        f"weight_ratio={args.silhouette_weight / args.negative_space_weight:.6g}",
        "",
        "Criteria:",
        f"  eval_interval={args.eval_interval}",
        f"  plateau_delta={args.plateau_delta}",
        f"  plateau_window={args.plateau_window}",
        f"  plateau_patience={args.plateau_patience}",
        f"  target_iou={args.target_iou}",
        f"  max_steps={args.max_steps}",
        f"  max_hours={args.max_hours}",
        "",
        "Result:",
        f"  stop_reason={stop_reason}",
        f"  setup_seconds={setup_seconds:.3f}",
        f"  total_seconds={total_seconds:.3f}",
    ]

    if plateau is not None:
        lines.extend([
            "  plateau_reached=True",
            f"  plateau_steps={plateau['step']}",
            f"  plateau_seconds={plateau['seconds']:.3f}",
            f"  plateau_mean_iou={plateau['mean_iou']:.6f}",
            f"  plateau_loss={plateau['loss']:.6f}",
            f"  plateau_patches={plateau['patches']}",
        ])
    else:
        lines.extend([
            "  plateau_reached=False",
            "  plateau_steps=-1",
            "  plateau_seconds=-1",
        ])

    if target is not None:
        lines.extend([
            "  target_iou_reached=True",
            f"  target_iou_steps={target['step']}",
            f"  target_iou_seconds={target['seconds']:.3f}",
            f"  target_iou_value={target['mean_iou']:.6f}",
            f"  target_iou_loss={target['loss']:.6f}",
            f"  target_iou_patches={target['patches']}",
        ])
    else:
        lines.extend([
            "  target_iou_reached=False",
            "  target_iou_steps=-1",
            "  target_iou_seconds=-1",
        ])

    if best is not None and final is not None:
        lines.extend([
            f"  best_mean_iou={best['mean_iou']:.6f}",
            f"  best_mean_iou_step={best['step']}",
            f"  final_step={final['step']}",
            f"  final_mean_iou={final['mean_iou']:.6f}",
            f"  final_loss={final['loss']:.6f}",
            f"  final_patches={final['patches']}",
        ])

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
    from core.swept_volume import SweptVolume

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

    srd_config = None
    if args.method == "srd":
        srd_config = {
            "enabled": True,
            "interval": args.srd_interval,
            "candidate_count": args.srd_candidates,
            "disable_swept_volume_adds": False,
            "loss_only_deletion": False,
            "disable_splitting": False,
        }

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
        srd_config=srd_config,
        swept_volume=swept_volume,
        silhouette_weight=args.silhouette_weight,
        negative_space_weight=args.negative_space_weight,
    )

    setup_seconds = time.perf_counter() - setup_started
    print(
        f"[Run] method={args.method}, seed={args.seed}, "
        f"max_steps={args.max_steps}, max_hours={args.max_hours}, "
        f"setup={setup_seconds:.1f}s",
        flush=True,
    )

    max_seconds = args.max_hours * 3600.0
    optimization_started = time.perf_counter()

    history: list[dict] = []
    plateau: dict | None = None
    target: dict | None = None
    sub_threshold_evals = 0
    stop_reason = "max_steps"

    for step_index in range(1, args.max_steps + 1):
        step_metrics = optimizer.step(step_index, args.max_steps)

        if step_index % args.eval_interval == 0 or step_index == 1:
            metrics, _, _ = optimizer.evaluate_snapshot()
            elapsed = time.perf_counter() - optimization_started
            entry = {
                "step": step_index,
                "seconds": elapsed,
                "mean_iou": float(metrics.get("mean_iou", 0.0)),
                "view1_iou": float(metrics.get("view1_iou", 0.0)),
                "view2_iou": float(metrics.get("view2_iou", 0.0)),
                "loss": float(metrics["loss"]),
                "patches": int(metrics["patches"]),
            }
            history.append(entry)

            if target is None and entry["mean_iou"] >= args.target_iou:
                target = entry
                print(
                    f"  [target] mean_iou={entry['mean_iou']:.6f} reached "
                    f"at step {entry['step']} ({entry['seconds']:.1f}s)",
                    flush=True,
                )

            if plateau is None:
                if _detect_plateau(history, args.plateau_window, args.plateau_delta):
                    sub_threshold_evals += 1
                    if sub_threshold_evals >= args.plateau_patience:
                        plateau = entry
                        stop_reason = "plateau"
                        print(
                            f"  [plateau] mean_iou={entry['mean_iou']:.6f} at "
                            f"step {entry['step']} ({entry['seconds']:.1f}s)",
                            flush=True,
                        )
                else:
                    sub_threshold_evals = 0

            if step_index % (args.eval_interval * 4) == 0 or step_index == 1:
                print(
                    f"  step {step_index}/{args.max_steps}: "
                    f"loss={entry['loss']:.6f}, mean_iou={entry['mean_iou']:.6f}, "
                    f"patches={entry['patches']}, {entry['seconds']:.1f}s",
                    flush=True,
                )

            if plateau is not None:
                break
            if elapsed >= max_seconds:
                stop_reason = "max_hours"
                print(f"  [stop] wall-clock cap reached at step {step_index}", flush=True)
                break

    total_seconds = time.perf_counter() - optimization_started

    _write_history_csv(output_dir / "iou_history.csv", history)
    _write_report(
        output_dir / "convergence.txt",
        args,
        started_at,
        history,
        plateau,
        target,
        stop_reason,
        total_seconds,
        setup_seconds,
    )
    print(f"\n[Convergence] report written to {output_dir / 'convergence.txt'}", flush=True)


if __name__ == "__main__":
    main()
