"""Send method-comparison experiments to a SLURM cluster.

Same machinery as submit_ablations.py, but the run settings compare
optimization methods instead of ablating SRD components:

    basic          plain gradient descent (no SRD)
    srd            full method with Stochastic Rewrite Descent
    annealing      simulated annealing, no SRD
    srd-annealing  SRD combined with simulated annealing

Examples
--------
    # Preview what would be submitted
    python submit_comparison.py --sweep-name methods --dry-run

    # basic vs srd vs annealing on every default pair, 3 trials each
    python submit_comparison.py --sweep-name methods --trials 3 --steps 2000 \
        --partition gpu --gres gpu:1 --env-setup "source .venv/bin/activate"

    # After jobs finish: aggregate every output.txt into one table
    python submit_comparison.py --sweep-name methods --collect
"""

from __future__ import annotations

from submit_ablations import run_cli

# Run setting -> extra run_ablation.py flags.
RUN_SETTINGS: dict[str, list[str]] = {
    "basic": ["--no-srd"],
    "srd": ["--mode", "full"],
    "annealing": ["--no-srd", "--simulated-annealing"],
    "srd-annealing": ["--mode", "full", "--simulated-annealing"],
}


def main() -> None:
    run_cli(
        RUN_SETTINGS,
        "Submit method-comparison experiments (basic / SRD / simulated "
        "annealing) to a SLURM cluster.",
        default_settings=["basic", "srd", "annealing"],
    )


if __name__ == "__main__":
    main()
