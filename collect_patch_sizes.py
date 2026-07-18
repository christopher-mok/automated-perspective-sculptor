"""Aggregate patch-size sweep results into one SRD-vs-base table.

The patchsize_* jobs write the compare_srd.py layout, not the
run_ablation.py one that ``submit_ablations.py --collect`` understands:

    <sweep-dir>/patchsize_<pair>_<n>/patches_<n>/comparison.txt

This walks every comparison.txt under a sweep directory, pulls the two
summary blocks (WITH SRD / WITHOUT SRD) out of each, and prints one row per
patch size so the scaling trend is readable at a glance.

Examples
--------
    # Table for one sweep
    python collect_patch_sizes.py results/slurm/20260718_0340

    # Same, as CSV for plotting
    python collect_patch_sizes.py results/slurm/20260718_0340 --csv sizes.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

# Summary keys read out of each block, in report order.
_SUMMARY_KEYS = ("completed_trials", "mean_final_loss", "std_final_loss", "mean_trial_seconds")


def _parse_config(text: str) -> dict[str, str]:
    """Pull the indented key=value lines from the Configuration block."""
    config: dict[str, str] = {}
    match = re.search(r"^Configuration:\n((?:[ \t]+\S.*\n)+)", text, re.MULTILINE)
    if not match:
        return config
    for line in match.group(1).splitlines():
        key, _, value = line.strip().partition("=")
        if value:
            config[key] = value.strip()
    return config


def _parse_summary(text: str, label: str) -> dict[str, float]:
    """Pull one 'Summary (WITH SRD):' style block into a float dict."""
    match = re.search(
        rf"^Summary \({re.escape(label)}\):\n((?:[ \t]+\S.*\n)+)",
        text,
        re.MULTILINE,
    )
    if not match:
        return {}

    summary: dict[str, float] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.strip().partition("=")
        if key in _SUMMARY_KEYS:
            try:
                summary[key] = float(value)
            except ValueError:
                pass
    return summary


def _pair_name(config: dict[str, str], report: Path) -> str:
    """Name the pair from the target images, falling back to the job dir."""
    stems = [
        Path(config[key]).stem
        for key in ("target1", "target2")
        if config.get(key) and config[key] != "None"
    ]
    if stems:
        return "_".join(stems)
    # <sweep>/patchsize_<pair>_<n>/patches_<n>/comparison.txt
    return re.sub(r"^patchsize_|_\d+$", "", report.parent.parent.name)


def _collect(sweep_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for report in sorted(sweep_dir.glob("**/comparison.txt")):
        text = report.read_text(encoding="utf-8")
        config = _parse_config(text)
        with_srd = _parse_summary(text, "WITH SRD")
        without_srd = _parse_summary(text, "WITHOUT SRD")

        try:
            n_patches = int(float(config.get("n_patches", "")))
        except ValueError:
            n_patches = -1

        srd_loss = with_srd.get("mean_final_loss")
        base_loss = without_srd.get("mean_final_loss")
        improvement = (
            (base_loss - srd_loss) / base_loss * 100.0
            if srd_loss is not None and base_loss not in (None, 0.0)
            else None
        )

        rows.append({
            "pair": _pair_name(config, report),
            "n_patches": n_patches,
            "srd_mean_loss": srd_loss,
            "base_mean_loss": base_loss,
            "improvement_pct": improvement,
            "srd_trials": with_srd.get("completed_trials"),
            "base_trials": without_srd.get("completed_trials"),
            "srd_std": with_srd.get("std_final_loss"),
            "base_std": without_srd.get("std_final_loss"),
            "srd_mean_seconds": with_srd.get("mean_trial_seconds"),
            "base_mean_seconds": without_srd.get("mean_trial_seconds"),
            "path": str(report.relative_to(sweep_dir)),
        })

    rows.sort(key=lambda row: (row["pair"], row["n_patches"]))
    return rows


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'pair':<22} {'patches':>7} {'srd_loss':>10} {'base_loss':>10} "
        f"{'improve%':>9} {'srd_s':>9} {'base_s':>9} {'trials':>8}"
    )
    print(header)
    print("-" * len(header))

    current_pair = None
    for row in rows:
        if current_pair is not None and row["pair"] != current_pair:
            print()
        current_pair = row["pair"]
        trials = (
            f"{_fmt(row['srd_trials'], '.0f')}/{_fmt(row['base_trials'], '.0f')}"
        )
        print(
            f"{row['pair']:<22} "
            f"{row['n_patches'] if row['n_patches'] >= 0 else '-':>7} "
            f"{_fmt(row['srd_mean_loss'], '.6f'):>10} "
            f"{_fmt(row['base_mean_loss'], '.6f'):>10} "
            f"{_fmt(row['improvement_pct'], '+.2f'):>9} "
            f"{_fmt(row['srd_mean_seconds'], '.1f'):>9} "
            f"{_fmt(row['base_mean_seconds'], '.1f'):>9} "
            f"{trials:>8}"
        )


def _print_incomplete(sweep_dir: Path, rows: list[dict]) -> None:
    """Name job dirs that produced no comparison.txt, so gaps aren't silent."""
    finished = {Path(row["path"]).parts[0] for row in rows}
    pending = sorted(
        job.name
        for job in sweep_dir.iterdir()
        if job.is_dir() and job.name not in finished
    )
    if pending:
        print(f"\n{len(pending)} job(s) with no comparison.txt yet (running or failed):")
        for name in pending:
            print(f"  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sweep_dir",
        help="Sweep directory to aggregate, e.g. results/slurm/20260718_0340.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Also write the rows to this CSV path.",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = _PROJECT_ROOT / sweep_dir
    if not sweep_dir.is_dir():
        sys.exit(f"Sweep directory not found: {sweep_dir}")

    rows = _collect(sweep_dir)
    if not rows:
        print(f"No comparison.txt files found under {sweep_dir}")
        _print_incomplete(sweep_dir, rows)
        return

    _print_table(rows)
    _print_incomplete(sweep_dir, rows)

    if args.csv:
        csv_path = Path(args.csv)
        fields = [key for key in rows[0] if key != "path"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} row(s) to {csv_path}")


if __name__ == "__main__":
    main()
