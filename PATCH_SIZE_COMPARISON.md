# Patch Size Comparison Scripts

Two new scripts for comparing SRD vs base approach at varying initial patch counts.

## Scripts Overview

### `compare_srd_patch_sizes.py`
Compares optimization for a single image pair across multiple initial patch sizes.

**Usage:**
```bash
python compare_srd_patch_sizes.py \
  --target1 images/horse.png \
  --target2 images/circle.png \
  --patch-sizes 5 10 20 30 \
  --trials 3 \
  --steps 1000
```

**Output Structure:**
```
comparisons/patch_sizes_<timestamp>/
├── SUMMARY.txt
├── patches_5/
│   ├── comparison.txt
│   ├── with_srd/
│   │   ├── with_srd.txt
│   │   ├── trial1_loss.csv
│   │   ├── trial2_loss.csv
│   │   └── trial3_loss.csv
│   └── without_srd/
│       └── [same structure]
├── patches_10/
│   └── [same structure]
├── patches_20/
│   └── [same structure]
└── patches_30/
    └── [same structure]
```

### `submit_patch_sizes.py`
Batch submission script that runs patch size comparisons for all 5 image pairs.

**Usage:**
```bash
python submit_patch_sizes.py
```

**Automatically runs comparisons for:**
- fire/water
- cat_face/bass
- horse/circle
- sun/moon
- axe/tree

**At patch sizes:** 5, 10, 20, 30
**With:** 3 trials, 1000 steps, 256 swept volume resolution

## Output Structure (Batch)

```
comparisons/patch_sizes_batch_<timestamp>/
├── BATCH_SUMMARY.txt              # Overall timing and status
├── fire_water/
│   ├── SUMMARY.txt
│   ├── patches_5/
│   ├── patches_10/
│   ├── patches_20/
│   └── patches_30/
├── cat_face_bass/
│   └── [same structure]
├── horse_circle/
│   └── [same structure]
├── sun_moon/
│   └── [same structure]
└── axe_tree/
    └── [same structure]
```

## Expected Results

The patch size comparisons allow analysis of:
1. **SRD scalability** - Does SRD benefit increase with problem size?
2. **Initialization quality** - How much does initial patch count affect final loss?
3. **Convergence behavior** - Do different sizes converge at different rates?
4. **Computational cost** - How does runtime scale with patch count?

## Key Metrics Collected

Per comparison:
- `final_loss` - Convergence quality
- `final_patches` - Final patch count after optimization
- `view1_iou` / `view2_iou` - Image coverage metrics
- `srd_total_adds` / `srd_total_deletes` - SRD activity
- Loss curves (CSV) - Convergence trajectory every 10 steps

## Configuration Defaults

| Parameter | Value |
|-----------|-------|
| Patch sizes | 5, 10, 20, 30 |
| Steps | 1000 |
| Trials | 3 per configuration |
| Swept volume resolution | 256 |
| Learning rate | 3.5e-3 |
| Device | cuda |

All other parameters use `compare_srd.py` defaults.

## Example Analysis

Compare SRD benefit across patch sizes:
```bash
# Run batch
python submit_patch_sizes.py

# Check BATCH_SUMMARY.txt for timing
cat comparisons/patch_sizes_batch_*/BATCH_SUMMARY.txt

# Compare results across patch sizes
for size in 5 10 20 30; do
  echo "=== Patch size $size ==="
  grep "SRD mean loss improvement" \
    comparisons/patch_sizes_batch_*/fire_water/patches_$size/comparison.txt
done
```

## Notes

- Each full batch (5 pairs × 4 sizes) takes ~25-30 hours
- Individual comparisons can be run independently
- IOU metrics included in all reports (requires target image with alpha channel)
- CSV loss curves enable detailed convergence analysis
