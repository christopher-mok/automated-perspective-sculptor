"""Run SRD vs base comparisons for multiple image pairs.

This script runs comparisons for multiple image pairs and saves results
organized by pair in a comparisons folder.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _run_comparison(
    pair_name: str,
    target1: str,
    target2: str,
    trials: int,
    steps: int,
    comparisons_dir: Path,
) -> bool:
    """Run a single comparison and return success status."""
    pair_output_dir = comparisons_dir / pair_name
    pair_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running comparison for: {pair_name}")
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
        "--output-dir", str(pair_output_dir),
        "--output-file", "comparison.txt",
        "--no-renders",
    ]

    result = subprocess.run(cmd, cwd=_PROJECT_ROOT)
    success = result.returncode == 0

    if success:
        print(f"[OK] Comparison for {pair_name} completed successfully")
    else:
        print(f"[ERROR] Comparison for {pair_name} failed with code {result.returncode}")

    return success


def main() -> None:
    # Image pairs to compare
    pairs = [
        ("fire_water", "images/fire.png", "images/water.png"),
        ("cat_face_bass", "images/cat_face.png", "images/bass.png"),
        ("horse_circle", "images/horse.png", "images/circle.png"),
        ("sun_moon", "images/sun.png", "images/moon.png"),
        ("axe_tree", "images/axe.png", "images/tree.png"),
    ]

    trials = 3
    steps = 1000

    started = datetime.now()
    comparisons_dir = _PROJECT_ROOT / "comparisons" / f"batch_{started.strftime('%Y%m%d_%H%M%S')}"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("SRD Comparison Batch Run")
    print(f"{'='*70}")
    print(f"Started: {started.isoformat(timespec='seconds')}")
    print(f"Pairs: {len(pairs)}")
    print(f"Trials per pair: {trials}")
    print(f"Steps per trial: {steps}")
    print(f"Output directory: {comparisons_dir}")
    print(f"{'='*70}\n")

    run_started = time.perf_counter()
    results = []

    for pair_name, target1, target2 in pairs:
        pair_started = time.perf_counter()
        success = _run_comparison(
            pair_name,
            target1,
            target2,
            trials,
            steps,
            comparisons_dir,
        )
        pair_time = time.perf_counter() - pair_started

        results.append({
            "pair": pair_name,
            "success": success,
            "time_seconds": pair_time,
        })

    total_time = time.perf_counter() - run_started

    # Write summary report
    summary_path = comparisons_dir / "SUMMARY.txt"
    summary_lines = [
        "SRD vs Base Comparison Batch Summary",
        f"Completed: {datetime.now().isoformat(timespec='seconds')}",
        f"Started: {started.isoformat(timespec='seconds')}",
        "",
        "Configuration:",
        f"  Pairs: {len(pairs)}",
        f"  Trials per pair: {trials}",
        f"  Steps per trial: {steps}",
        "",
        "Results:",
    ]

    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        summary_lines.append(
            f"  {result['pair']:20s} {status:7s} ({result['time_seconds']:7.1f}s)"
        )

    successful = sum(1 for r in results if r["success"])
    summary_lines.extend([
        "",
        f"Completed pairs: {successful}/{len(pairs)}",
        f"Total time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)",
        "",
        "Output structure:",
        "  comparisons/batch_<timestamp>/",
        "    SUMMARY.txt",
        "    fire_water/",
        "      comparison.txt",
        "      with_srd/",
        "        with_srd.txt",
        "        trial1_loss.csv",
        "        trial2_loss.csv",
        "        trial3_loss.csv",
        "      without_srd/",
        "        without_srd.txt",
        "        trial1_loss.csv",
        "        trial2_loss.csv",
        "        trial3_loss.csv",
        "    cat_face_bass/",
        "      ...",
        "    [etc for other pairs]",
    ])

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"\n{'='*70}")
    print("Batch Summary")
    print(f"{'='*70}")
    print(f"Successful: {successful}/{len(pairs)}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
    print(f"Summary report: {summary_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
