# Comparison and Ablation Commands

## SRD Comparison (`compare_srd.py`)

This command compares optimization performance with SRD enabled vs disabled using the same image pair.

### Key Features:
- Runs N trials with SRD enabled (full approach)
- Runs N trials without SRD (base approach: random initialization, no swept volume)
- Fixed step count and controlled randomness for fair comparison
- Exports loss history as CSV every 10 steps for each trial
- Generates comprehensive comparison report

### Usage:

```bash
python compare_srd.py \
  --target1 images/horse.png \
  --target2 images/circle.png \
  --trials 3 \
  --steps 500 \
  --output-file my_comparison.txt
```

### Output Files:

```
results/srd_comparison_<timestamp>/
├── my_comparison.txt              # Main comparison report
├── with_srd/
│   ├── with_srd.txt              # Full run_ablation report (with SRD)
│   ├── trial1_loss.csv           # Loss values: step, loss (every 10 steps)
│   ├── trial2_loss.csv
│   └── trial3_loss.csv
└── without_srd/
    ├── without_srd.txt           # Full run_ablation report (without SRD)
    ├── trial1_loss.csv           # Loss values: step, loss (every 10 steps)
    ├── trial2_loss.csv
    └── trial3_loss.csv
```

### Example Output:

```
SRD mean loss improvement: 6.38%
WITH SRD:    mean loss = 0.886109, final patches = 13
WITHOUT SRD: mean loss = 0.946537, final patches = 20
Absolute difference: 0.060428
```

### Common Options:

- `--trials N`: Number of trials per configuration (default: 3)
- `--steps N`: Optimization steps per trial (default: 500)
- `--device cuda`: Use CUDA (or cpu)
- `--n-patches N`: Initial patch count (default: 20)
- `--lr VALUE`: Learning rate (default: 0.003)
- `--output-file NAME`: Custom report filename (default: output.txt)
- `--no-renders`: Skip saving render images
- `--output-dir DIR`: Specify output directory

---

## Ablation Study (`run_ablation.py`)

This command runs optimization with various configuration options and exports per-trial loss histories.

### Usage:

```bash
python run_ablation.py \
  --target1 images/horse.png \
  --target2 images/circle.png \
  --trials 3 \
  --steps 500 \
  --output-file ablation_results.txt
```

### Modes:

- **full** (default): Complete method with swept-volume-guided additions, rule-based deletion, splitting
- **ablation**: Disables all features above

### Ablation Flags:

- `--no-swept-volume-adds`: Disable swept-volume-guided SRD additions
- `--loss-only-deletion`: Delete only when loss improves
- `--no-splitting`: Disable splitting rewrites
- `--no-srd`: Disable SRD entirely

### Output Files:

```
results/full_<timestamp>/
├── output.txt         # Main report (or custom name via --output-file)
├── trial1_loss.csv   # Loss values: step, loss
├── trial2_loss.csv
├── trial3_loss.csv
├── trial1_view1.png  # Final render for view 1 (unless --no-renders)
├── trial2_view1.png
└── ...
```

### CSV Format:

```csv
step,loss
1,1.187894
10,1.057890
20,0.953717
```

---

## Base vs SRD Comparison Details

The `compare_srd.py` command runs two distinct optimization approaches:

### WITH SRD (Full Approach):
- Uses swept volume for guided patch initialization
- Stochastic rasterization derivatives (SRD) for intelligent rewriting
- Rule-based patch addition and deletion
- Patch splitting rewrites

### WITHOUT SRD (Base Approach):
- Random patch initialization only (no swept volume)
- No SRD operations
- Loss-only deletion (patches deleted only when improving loss)
- No splitting rewrites

This provides a direct comparison of the SRD contribution to optimization quality.
