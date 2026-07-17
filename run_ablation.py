"""Headless CLI for full-method and ablation optimization runs.

Two run settings are provided:

    full      The complete method (swept-volume-guided SRD additions,
              rule-based deletion, splitting).
    ablation  Disables swept-volume-guided additions, rule-based deletion
              (pieces are only deleted when doing so improves the loss),
              and splitting.

Each run writes an ``output.txt`` report (time taken, final loss, etc.)
into the output directory, with one entry per trial when ``--trials`` > 1.

Examples
--------
    python run_ablation.py --target1 images/horse.png --target2 images/circle.png
    python run_ablation.py --target1 images/horse.png --target2 images/circle.png \
        --mode ablation --trials 3 --steps 500
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent
_TARGET_TRANSPARENT_BORDER_FRACTION = 0.10
_SCENE_CAMERA_FOV_DEG = 30.22  # matches ui/main_window.py
_HANGING_PLANE_Y = 3.5


def _make_scene_cameras():
    """Two scene cameras 90° apart (same setup as the UI)."""
    from scene.camera import Camera

    radius = 8.0
    origin = np.zeros(3, dtype=np.float32)
    shared = dict(
        target=origin,
        fov=_SCENE_CAMERA_FOV_DEG,
        aspect=4.0 / 3.0,
        near=0.35,
        far=18.0,
    )
    cam1 = Camera(
        position=np.array([0.0, 0.0, radius], dtype=np.float32),
        label="View 1",
        **shared,
    )
    cam2 = Camera(
        position=np.array([radius, 0.0, 0.0], dtype=np.float32),
        label="View 2",
        **shared,
    )
    return [cam1, cam2]


def _load_target_image_with_border(path: str) -> np.ndarray:
    """Load an image as RGBA and add a transparent border (same as the UI)."""
    from PIL import Image, ImageOps

    image = Image.open(path).convert("RGBA")
    border = max(1, int(round(max(image.size) * _TARGET_TRANSPARENT_BORDER_FRACTION)))
    return np.array(ImageOps.expand(image, border=border, fill=(0, 0, 0, 0)))


def _save_render(render, path: Path) -> None:
    from PIL import Image

    array = np.asarray(render)
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(array, mode="RGBA").save(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-method or ablation optimization headlessly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", default=None, help="View 2 target image path.")
    parser.add_argument(
        "--mode",
        choices=("full", "ablation"),
        default="full",
        help="Run setting: 'full' method or 'ablation' (disables swept-volume "
             "additions, rule-based deletion, and splitting).",
    )
    parser.add_argument(
        "--no-swept-volume-adds",
        action="store_true",
        help="Ablate only the swept-volume-guided SRD additions.",
    )
    parser.add_argument(
        "--loss-only-deletion",
        action="store_true",
        help="Ablate only the deletion rules (delete solely when loss improves).",
    )
    parser.add_argument(
        "--no-splitting",
        action="store_true",
        help="Ablate only the splitting rewrites.",
    )
    parser.add_argument("--trials", type=int, default=1, help="Number of trials.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Explicit per-trial seeds (overrides --trials count).",
    )
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps per trial.")
    parser.add_argument("--n-patches", type=int, default=20, help="Initial patch count.")
    parser.add_argument("--lr", type=float, default=3.5e-3, help="Learning rate.")
    parser.add_argument("--device", default="cpu", help="PyTorch device string.")
    parser.add_argument("--palette", default="", help="Palette, e.g. '#111111,#f4d35e'.")
    parser.add_argument("--hanging-plane-size", type=float, default=5.0)
    parser.add_argument("--swept-resolution", type=int, default=108)
    parser.add_argument("--srd-interval", type=int, default=50)
    parser.add_argument("--srd-candidates", type=int, default=32)
    parser.add_argument("--no-srd", action="store_true", help="Disable SRD entirely.")
    parser.add_argument(
        "--simulated-annealing",
        action="store_true",
        help="Enable simulated-annealing parameter noise.",
    )
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=1.0,
        help="Starting temperature for simulated annealing.",
    )
    parser.add_argument(
        "--no-renders",
        action="store_true",
        help="Skip saving final per-trial render images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Report/render directory (default: results/<mode>_<timestamp>).",
    )
    parser.add_argument(
        "--output-file",
        default="output.txt",
        help="Output report filename (default: output.txt).",
    )
    return parser.parse_args()


def _save_loss_csv(output_path: Path, trial_number: int, loss_history: list[tuple[int, float]]) -> None:
    """Save loss over steps as CSV for a trial."""
    lines = ["step,loss"]
    for step, loss in loss_history:
        lines.append(f"{step},{loss:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trial_report_lines(trial: dict) -> list[str]:
    metrics = trial["metrics"]
    lines = [
        f"Trial {trial['trial']} (seed={trial['seed']}):",
        f"  time_seconds={trial['time_seconds']:.3f}",
        f"  optimization_seconds={trial['optimization_seconds']:.3f}",
        f"  average_step_seconds={trial['average_step_seconds']:.6f}",
        f"  final_loss={metrics['loss']:.6f}",
        f"  final_patches={int(metrics['patches'])}",
        f"  view1_mse={metrics['view1_mse']:.6f}",
        f"  view2_loss={metrics['view2_loss']:.6f}",
        f"  view1_silhouette={metrics['view1_silhouette']:.6f}",
        f"  view2_silhouette={metrics['view2_silhouette']:.6f}",
        f"  view1_negative_space={metrics['view1_negative_space']:.6f}",
        f"  view2_negative_space={metrics['view2_negative_space']:.6f}",
        f"  overlap={metrics['overlap']:.6f}",
        f"  srd_total_adds={trial['srd_total_adds']}",
        f"  srd_total_deletes={trial['srd_total_deletes']}",
        f"  final_pass_tiny_deleted={trial['final_tiny_deleted']}",
        f"  final_pass_loss_improving_deleted={trial['final_loss_improving_deleted']}",
    ]
    return lines


def _write_report(
    output_path: Path,
    args: argparse.Namespace,
    ablation_flags: dict[str, bool],
    seeds: list[int],
    started_at: str,
    trials: list[dict],
    total_seconds: float | None,
) -> None:
    lines = [
        "Perspective Sculptor run report",
        f"started={started_at}",
        f"mode={args.mode}",
        f"srd_enabled={not args.no_srd}",
        f"simulated_annealing={args.simulated_annealing}",
        f"disable_swept_volume_adds={ablation_flags['disable_swept_volume_adds']}",
        f"loss_only_deletion={ablation_flags['loss_only_deletion']}",
        f"disable_splitting={ablation_flags['disable_splitting']}",
        f"target1={args.target1}",
        f"target2={args.target2}",
        f"steps={args.steps}",
        f"trials={len(seeds)}",
        f"seeds={','.join(str(seed) for seed in seeds)}",
        f"n_patches={args.n_patches}",
        f"learning_rate={args.lr:.6g}",
        f"device={args.device}",
        f"palette={args.palette!r}",
        f"hanging_plane_size={args.hanging_plane_size:.3f}",
        f"swept_volume_resolution={args.swept_resolution}",
        "",
    ]
    for trial in trials:
        lines.extend(_trial_report_lines(trial))
        lines.append("")

    if trials:
        losses = [trial["metrics"]["loss"] for trial in trials]
        times = [trial["time_seconds"] for trial in trials]
        best = min(trials, key=lambda trial: trial["metrics"]["loss"])
        lines.extend([
            "Summary:",
            f"  completed_trials={len(trials)}",
            f"  mean_final_loss={float(np.mean(losses)):.6f}",
            f"  std_final_loss={float(np.std(losses)):.6f}",
            f"  best_final_loss={best['metrics']['loss']:.6f} (trial {best['trial']})",
            f"  mean_trial_seconds={float(np.mean(times)):.3f}",
        ])
        if total_seconds is not None:
            lines.append(f"  total_seconds={total_seconds:.3f}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    ablation_flags = {
        "disable_swept_volume_adds": args.mode == "ablation" or args.no_swept_volume_adds,
        "loss_only_deletion": args.mode == "ablation" or args.loss_only_deletion,
        "disable_splitting": args.mode == "ablation" or args.no_splitting,
    }
    seeds = args.seeds if args.seeds else list(range(max(1, args.trials)))

    started = datetime.now()
    started_at = started.isoformat(timespec="seconds")
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            _PROJECT_ROOT / "results"
            / f"{args.mode}_{started.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.output_file

    import torch

    from core.initialization import initialize_patches
    from core.optimizer import SceneOptimizer
    from core.swept_volume import SweptVolume

    cameras = _make_scene_cameras()
    target1 = _load_target_image_with_border(args.target1)
    target2 = (
        _load_target_image_with_border(args.target2)
        if args.target2 is not None else None
    )

    swept_volume = None
    if target2 is not None:
        print(f"[Swept volume] building (resolution={args.swept_resolution})...")
        swept_volume = SweptVolume.from_images(
            target1,
            target2,
            cameras,
            hanging_plane_size=args.hanging_plane_size,
            resolution=args.swept_resolution,
        )

    srd_config: dict[str, object] | None = None
    if not args.no_srd:
        srd_config = {
            "enabled": True,
            "interval": args.srd_interval,
            "candidate_count": args.srd_candidates,
            **ablation_flags,
        }

    print(
        f"[Run] mode={args.mode}, trials={len(seeds)}, seeds={seeds}, "
        f"steps={args.steps}, srd={not args.no_srd}, ablation_flags={ablation_flags}"
    )

    run_started = time.perf_counter()
    trials: list[dict] = []
    for trial_number, seed in enumerate(seeds, start=1):
        print(f"[Trial {trial_number}/{len(seeds)}] seed={seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        trial_started = time.perf_counter()
        patches = initialize_patches(
            mode="Experimental",
            n_patches=args.n_patches,
            reference_image=target1,
            cameras=cameras,
            device=args.device,
            seed=seed,
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
            simulated_annealing=args.simulated_annealing,
            initial_temperature=args.initial_temperature,
            swept_volume=swept_volume,
        )

        optimization_started = time.perf_counter()
        loss_history = []
        for step_index in range(1, args.steps + 1):
            step_metrics = optimizer.step(step_index, args.steps)
            if step_index % 10 == 0 or step_index == 1:
                loss_history.append((step_index, float(step_metrics['loss'])))
            if step_index % 50 == 0 or step_index == args.steps:
                print(
                    f"  step {step_index}/{args.steps}: "
                    f"loss={step_metrics['loss']:.6f}, "
                    f"patches={int(step_metrics['patches'])}"
                )
        optimization_seconds = time.perf_counter() - optimization_started

        final_deletion_stats = {"tiny_deleted": 0.0, "loss_improving_deleted": 0.0}
        if optimizer.srd is not None and optimizer.srd.enabled:
            final_deletion_stats.update(optimizer.srd.final_deletion_pass(
                optimizer,
                optimizer.optim,
                args.steps,
            ))

        metrics, render1, render2 = optimizer.evaluate_snapshot()
        if not args.no_renders:
            _save_render(render1, output_dir / f"trial{trial_number}_view1.png")
            _save_render(render2, output_dir / f"trial{trial_number}_view2.png")

        srd_stats = optimizer.srd.stats if optimizer.srd is not None else None
        trial = {
            "trial": trial_number,
            "seed": seed,
            "time_seconds": time.perf_counter() - trial_started,
            "optimization_seconds": optimization_seconds,
            "average_step_seconds": optimization_seconds / max(1, args.steps),
            "metrics": metrics,
            "srd_total_adds": srd_stats.total_added if srd_stats else 0,
            "srd_total_deletes": srd_stats.total_deleted if srd_stats else 0,
            "final_tiny_deleted": int(final_deletion_stats["tiny_deleted"]),
            "final_loss_improving_deleted": int(
                final_deletion_stats["loss_improving_deleted"]
            ),
        }
        trials.append(trial)

        # Save loss history CSV
        csv_path = output_dir / f"trial{trial_number}_loss.csv"
        _save_loss_csv(csv_path, trial_number, loss_history)
        print(
            f"[Trial {trial_number}] done: loss={metrics['loss']:.6f}, "
            f"time={trial['time_seconds']:.1f}s"
        )

        # Refresh the report after every trial so long runs stay inspectable.
        _write_report(
            report_path, args, ablation_flags, seeds, started_at, trials,
            total_seconds=None,
        )

    _write_report(
        report_path, args, ablation_flags, seeds, started_at, trials,
        total_seconds=time.perf_counter() - run_started,
    )
    print(f"[Run] report written to {report_path}")


if __name__ == "__main__":
    main()
