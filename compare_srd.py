"""Compare optimization performance with and without SRD using the same image pair.

This script runs the same number of trials with SRD enabled and disabled,
then outputs a comparison report to output.txt.

Example:
    python compare_srd.py --target1 images/horse.png --target2 images/circle.png \
        --trials 3 --steps 500
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SRD-enabled vs SRD-disabled optimization on the same image pair.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", default=None, help="View 2 target image path.")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration.")
    parser.add_argument("--steps", type=int, default=500, help="Optimization steps per trial.")
    parser.add_argument("--n-patches", type=int, default=20, help="Initial patch count.")
    parser.add_argument("--lr", type=float, default=3.5e-3, help="Learning rate.")
    parser.add_argument("--device", default="cuda", help="PyTorch device string.")
    parser.add_argument("--palette", default="", help="Palette, e.g. '#111111,#f4d35e'.")
    parser.add_argument("--hanging-plane-size", type=float, default=5.0)
    parser.add_argument("--swept-resolution", type=int, default=108)
    parser.add_argument("--srd-interval", type=int, default=50)
    parser.add_argument("--srd-candidates", type=int, default=32)
    parser.add_argument("--no-renders", action="store_true", help="Skip saving render images.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/srd_comparison_<timestamp>).",
    )
    parser.add_argument(
        "--output-file",
        default="output.txt",
        help="Output report filename (default: output.txt).",
    )
    return parser.parse_args()


def _run_trials(
    enable_srd: bool,
    trials: int,
    steps: int,
    target1: str,
    target2: str | None,
    n_patches: int,
    lr: float,
    device: str,
    palette: str,
    hanging_plane_size: float,
    swept_resolution: int,
    srd_interval: int,
    srd_candidates: int,
    output_dir: Path,
    no_renders: bool,
    output_file: str,
) -> list[dict]:
    """Run trials using run_ablation.py and parse results."""
    mode_label = "with_srd" if enable_srd else "without_srd"
    mode_output_dir = output_dir / mode_label
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "run_ablation.py"),
        "--target1", target1,
        "--trials", str(trials),
        "--steps", str(steps),
        "--n-patches", str(n_patches),
        "--lr", str(lr),
        "--device", device,
        "--hanging-plane-size", str(hanging_plane_size),
        "--swept-resolution", str(swept_resolution),
        "--srd-interval", str(srd_interval),
        "--srd-candidates", str(srd_candidates),
        "--output-dir", str(mode_output_dir),
        "--output-file", output_file,
    ]

    if target2 is not None:
        cmd.extend(["--target2", target2])
    if not enable_srd:
        cmd.append("--no-srd")
        # Base approach: disable swept volume and use random initialization only
        cmd.append("--no-swept-volume-adds")
        cmd.append("--loss-only-deletion")
    if palette:
        cmd.extend(["--palette", palette])
    if no_renders:
        cmd.append("--no-renders")

    print(f"\n{'='*70}")
    print(f"Running {trials} trials {'WITH' if enable_srd else 'WITHOUT'} SRD")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[ERROR] run_ablation.py failed with code {result.returncode}")
        return []

    # Parse the output file from run_ablation
    report_file = mode_output_dir / output_file
    if not report_file.exists():
        print(f"[ERROR] Report file not found: {report_file}")
        return []

    trials_data = _parse_report_file(report_file, enable_srd)
    return trials_data


def _parse_report_file(report_path: Path, enable_srd: bool) -> list[dict]:
    """Parse output.txt from run_ablation.py to extract trial metrics."""
    content = report_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    trials_data = []
    current_trial = None

    for line in lines:
        line = line.strip()

        if line.startswith("Trial"):
            if current_trial is not None:
                trials_data.append(current_trial)
            # Extract trial number and seed from "Trial X (seed=Y):"
            parts = line.split("(seed=")
            trial_num = parts[0].split()[-1]
            seed = parts[1].rstrip("):") if len(parts) > 1 else "0"
            current_trial = {
                "srd_enabled": enable_srd,
                "trial": int(trial_num),
                "seed": int(seed),
                "metrics": {},
            }
        elif current_trial is not None and "=" in line and line.endswith(":") is False:
            # Parse metric lines like "  final_loss=0.123456"
            key, value = line.split("=", 1)
            key = key.strip()
            try:
                # Try to parse as float first, then int, then string
                try:
                    value = float(value)
                except ValueError:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                current_trial["metrics"][key] = value
            except Exception:
                pass

    if current_trial is not None:
        trials_data.append(current_trial)

    return trials_data


def _write_comparison_report(
    output_path: Path,
    args: argparse.Namespace,
    with_srd_trials: list[dict],
    without_srd_trials: list[dict],
    total_seconds: float,
) -> None:
    """Write a comprehensive comparison report."""
    lines = [
        "SRD Comparison Report",
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        "",
        "Configuration:",
        f"  target1={args.target1}",
        f"  target2={args.target2}",
        f"  trials_per_config={args.trials}",
        f"  steps={args.steps}",
        f"  n_patches={args.n_patches}",
        f"  learning_rate={args.lr:.6g}",
        f"  device={args.device}",
        f"  hanging_plane_size={args.hanging_plane_size}",
        f"  swept_volume_resolution={args.swept_resolution}",
        f"  srd_interval={args.srd_interval}",
        f"  srd_candidates={args.srd_candidates}",
        "",
        "=" * 70,
        "WITH SRD ENABLED",
        "=" * 70,
    ]

    if with_srd_trials:
        for trial in with_srd_trials:
            lines.append(f"\nTrial {trial['trial']} (seed={trial['seed']}):")
            for key, value in trial["metrics"].items():
                if isinstance(value, float):
                    lines.append(f"  {key}={value:.6f}")
                else:
                    lines.append(f"  {key}={value}")

        with_srd_losses = [t["metrics"].get("final_loss", float("nan")) for t in with_srd_trials]
        with_srd_times = [
            t["metrics"].get("time_seconds", 0) for t in with_srd_trials
        ]
        lines.extend([
            "",
            "Summary (WITH SRD):",
            f"  completed_trials={len(with_srd_trials)}",
            f"  mean_final_loss={float(np.mean(with_srd_losses)):.6f}",
            f"  std_final_loss={float(np.std(with_srd_losses)):.6f}",
            f"  best_final_loss={float(np.min(with_srd_losses)):.6f}",
            f"  worst_final_loss={float(np.max(with_srd_losses)):.6f}",
            f"  mean_trial_seconds={float(np.mean(with_srd_times)):.3f}",
        ])
    else:
        lines.append("\n[No trials data for WITH SRD]")

    lines.extend([
        "",
        "=" * 70,
        "WITHOUT SRD",
        "=" * 70,
    ])

    if without_srd_trials:
        for trial in without_srd_trials:
            lines.append(f"\nTrial {trial['trial']} (seed={trial['seed']}):")
            for key, value in trial["metrics"].items():
                if isinstance(value, float):
                    lines.append(f"  {key}={value:.6f}")
                else:
                    lines.append(f"  {key}={value}")

        without_srd_losses = [t["metrics"].get("final_loss", float("nan")) for t in without_srd_trials]
        without_srd_times = [
            t["metrics"].get("time_seconds", 0) for t in without_srd_trials
        ]
        lines.extend([
            "",
            "Summary (WITHOUT SRD):",
            f"  completed_trials={len(without_srd_trials)}",
            f"  mean_final_loss={float(np.mean(without_srd_losses)):.6f}",
            f"  std_final_loss={float(np.std(without_srd_losses)):.6f}",
            f"  best_final_loss={float(np.min(without_srd_losses)):.6f}",
            f"  worst_final_loss={float(np.max(without_srd_losses)):.6f}",
            f"  mean_trial_seconds={float(np.mean(without_srd_times)):.3f}",
        ])
    else:
        lines.append("\n[No trials data for WITHOUT SRD]")

    # Comparison
    if with_srd_trials and without_srd_trials:
        with_srd_losses = np.array([t["metrics"].get("final_loss", float("nan")) for t in with_srd_trials])
        without_srd_losses = np.array([t["metrics"].get("final_loss", float("nan")) for t in without_srd_trials])

        mean_with = float(np.mean(with_srd_losses))
        mean_without = float(np.mean(without_srd_losses))
        improvement = ((mean_without - mean_with) / mean_without * 100) if mean_without != 0 else 0

        lines.extend([
            "",
            "=" * 70,
            "COMPARISON",
            "=" * 70,
            f"  SRD mean loss improvement: {improvement:.2f}%",
            f"  (negative = SRD is worse)",
            "",
            f"  WITH SRD:    mean loss = {mean_with:.6f}",
            f"  WITHOUT SRD: mean loss = {mean_without:.6f}",
            f"  Absolute difference: {abs(mean_with - mean_without):.6f}",
        ])

    lines.extend([
        "",
        f"Total runtime: {total_seconds:.1f} seconds",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()

    started = datetime.now()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            _PROJECT_ROOT / "results"
            / f"srd_comparison_{started.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.output_file

    run_started = time.perf_counter()

    # Run with SRD
    with_srd_trials = _run_trials(
        enable_srd=True,
        trials=args.trials,
        steps=args.steps,
        target1=args.target1,
        target2=args.target2,
        n_patches=args.n_patches,
        lr=args.lr,
        device=args.device,
        palette=args.palette,
        hanging_plane_size=args.hanging_plane_size,
        swept_resolution=args.swept_resolution,
        srd_interval=args.srd_interval,
        srd_candidates=args.srd_candidates,
        output_dir=output_dir,
        no_renders=args.no_renders,
        output_file="with_srd.txt",
    )

    # Run without SRD
    without_srd_trials = _run_trials(
        enable_srd=False,
        trials=args.trials,
        steps=args.steps,
        target1=args.target1,
        target2=args.target2,
        n_patches=args.n_patches,
        lr=args.lr,
        device=args.device,
        palette=args.palette,
        hanging_plane_size=args.hanging_plane_size,
        swept_resolution=args.swept_resolution,
        srd_interval=args.srd_interval,
        srd_candidates=args.srd_candidates,
        output_dir=output_dir,
        no_renders=args.no_renders,
        output_file="without_srd.txt",
    )

    total_seconds = time.perf_counter() - run_started

    _write_comparison_report(
        report_path,
        args,
        with_srd_trials,
        without_srd_trials,
        total_seconds,
    )
    print(f"\n[Comparison] report written to {report_path}")


if __name__ == "__main__":
    main()
