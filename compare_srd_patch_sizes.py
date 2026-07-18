"""Compare SRD vs base at varying initial patch sizes.

This script compares optimization with different initial patch counts
to study how SRD benefit scales with problem complexity.

Example:
    python compare_srd_patch_sizes.py --target1 images/horse.png \
        --target2 images/circle.png --patch-sizes 5 10 20 30 --steps 1000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _run_comparison(
    patch_size: int,
    target1: str,
    target2: str,
    trials: int,
    steps: int,
    comparisons_dir: Path,
) -> bool:
    """Run a single comparison for a given patch size."""
    size_output_dir = comparisons_dir / f"patches_{patch_size}"
    size_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running comparison with {patch_size} initial patches")
    print(f"  Target 1: {target1}")
    print(f"  Target 2: {target2}")
    print(f"  Trials: {trials}, Steps: {steps}")
    print(f"{'='*70}\n")

    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "compare_srd.py"),
        "--target1", target1,
        "--target2", target2,
        "--trials", str(trials),
        "--steps", str(steps),
        "--n-patches", str(patch_size),
        "--output-dir", str(size_output_dir),
        "--output-file", "comparison.txt",
        "--no-renders",
    ]

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    success = result.returncode == 0

    if success:
        print(f"[OK] Comparison with {patch_size} patches completed successfully")
    else:
        print(f"[ERROR] Comparison with {patch_size} patches failed with code {result.returncode}")

    return success


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SRD vs base approach at varying initial patch sizes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target1", required=True, help="View 1 target image path.")
    parser.add_argument("--target2", default=None, help="View 2 target image path.")
    parser.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30],
        help="Initial patch sizes to test.",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration.")
    parser.add_argument("--steps", type=int, default=1000, help="Optimization steps per trial.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/srd_patch_comparison_<timestamp>).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    started = datetime.now()
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            _PROJECT_ROOT / "comparisons"
            / f"patch_sizes_{started.strftime('%Y%m%d_%H%M%S')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SRD Patch Size Comparison")
    print(f"{'='*70}")
    print(f"Started: {started.isoformat(timespec='seconds')}")
    print(f"Patch sizes: {args.patch_sizes}")
    print(f"Trials per size: {args.trials}")
    print(f"Steps per trial: {args.steps}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")

    run_started = time.perf_counter()
    results = []

    for patch_size in args.patch_sizes:
        size_started = time.perf_counter()
        success = _run_comparison(
            patch_size,
            args.target1,
            args.target2,
            args.trials,
            args.steps,
            output_dir,
        )
        size_time = time.perf_counter() - size_started

        results.append({
            "patch_size": patch_size,
            "success": success,
            "time_seconds": size_time,
        })

    total_time = time.perf_counter() - run_started

    # Write summary report
    summary_path = output_dir / "SUMMARY.txt"
    summary_lines = [
        "SRD Patch Size Comparison Summary",
        f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"Started: {started.isoformat(timespec='seconds')}",
        "",
        "Configuration:",
        f"  Target 1: {args.target1}",
        f"  Target 2: {args.target2}",
        f"  Patch sizes tested: {args.patch_sizes}",
        f"  Trials per size: {args.trials}",
        f"  Steps per trial: {args.steps}",
        "",
        "Results:",
    ]

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        summary_lines.append(
            f"  patches={result['patch_size']:2d}  {status:7s} ({result['time_seconds']:7.1f}s)"
        )

    successful = sum(1 for r in results if r["success"])
    summary_lines.extend([
        "",
        f"Completed sizes: {successful}/{len(args.patch_sizes)}",
        f"Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)",
        "",
        "Output structure:",
        "  comparisons/patch_sizes_<timestamp>/",
        "    SUMMARY.txt",
        "    patches_5/",
        "      comparison.txt",
        "      with_srd/",
        "        with_srd.txt",
        "        trial1_loss.csv, trial2_loss.csv, trial3_loss.csv",
        "      without_srd/",
        "        without_srd.txt",
        "        trial1_loss.csv, trial2_loss.csv, trial3_loss.csv",
        "    patches_10/",
        "      ...",
        "    [etc for other sizes]",
    ])

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\n{'='*70}")
    print("Comparison Summary")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(args.patch_sizes)}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
    print(f"Summary report: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
