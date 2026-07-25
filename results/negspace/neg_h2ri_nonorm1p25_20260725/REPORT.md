# hinge2 + restart + importance deletion — `neg_h2ri_nonorm1p25_20260725`

Arm `hinge2_restart_importnet`: the fit-then-shed count objective
(`--count-objective`, count_lambda 500, count_target_iou 0.91, FULL_STACK
scheduling) with conflict-gated restart **and** net deletion-importance
sampling. First run of all three together — restart and importance deletion had
only ever been measured separately, on top of plain SRD.

Weights `nonorm1p25`: silhouette 2.0, negative space 2.5, non-normalized (the
original weights). Swept resolution 300, 64 SRD candidates,
`--srd-min-patch-area 0.005`, `--lambda-count 0`, 20 starting patches, planar
overlap, constrained theta, seed 0, 20000-step ceiling with plateau stop at min
1500 / patience 300.

Collected 2026-07-25 with `--collect`. **7 of 8 jobs finished**; horse_circle
(4281934) was still running at 7h06m and is not included.

## Final IoU and final piece count

| pair | final IoU | best IoU | final pieces | start | max | min | step | stop |
|---|---|---|---|---|---|---|---|---|
| water_fire | 0.9353 | 0.9375 | **9** | 20 | 29 | 9 | 1500 | shed_converged |
| cat_face_bass | 0.9318 | 0.9481 | **10** | 20 | 31 | 10 | 1500 | shed_converged |
| robot_man | 0.9151 | 0.9384 | **11** | 20 | 38 | 11 | 2400 | shed_converged |
| sun_moon | 0.9139 | 0.9439 | **18** | 20 | 29 | 18 | 2600 | shed_converged |
| acm_scf | 0.8913 | 0.8951 | **44** | 20 | 44 | 20 | 1875 | converged |
| dance_argument | 0.8767 | 0.8800 | **29** | 20 | 30 | 20 | 1525 | converged |
| teapot_droplets | 0.7987 | 0.8212 | **35** | 20 | 36 | 20 | 1800 | converged |
| **mean (n=7)** | **0.8947** | 0.9092 | **22.3** | | | | | |

## The runs split into two populations

The `stop_reason` column separates them cleanly, and piece count follows:

**Shed converged (4 runs)** — the count objective reached its IoU target, entered
the shedding phase, and drove piece count *below* the 20 it started with:
water_fire to 9, cat_face_bass to 10, robot_man to 11, sun_moon to 18. Mean
final IoU 0.924 at a mean of 12 pieces. Every one of these peaked well above 20
first (max 29–38) and then shed back down, which is the fit-then-shed schedule
working as designed.

**Plain converged (3 runs)** — never hit the target IoU, so shedding never
started and the count objective never bit. They plateaued with piece count at or
near their maximum: acm_scf 44 (its own max), teapot_droplets 35, dance_argument
29. Mean final IoU 0.856 at a mean of 36 pieces.

So the piece-count mean of 22.3 is not describing any actual run — it averages a
9–18 cluster against a 29–44 cluster. The three hard pairs are the same ones
that lag in every other batch.

Worth noting acm_scf ends at 44 pieces, more than double its start, having never
shed a single piece.

## Caveat on attribution

This arm changes three things at once relative to plain SRD. Nothing here
separates the count objective from restart from importance deletion — if these
numbers look good or bad, the cause is not identifiable from this batch alone.
The intermediate arms (`hinge2`, `hinge2_restart`, `importnet`) exist in
`submit_negspace.py` and would need to be run at these same settings to
attribute it.

## Files

- `collected/summary.tsv` — all 27 metrics per run
- `collected/curves.csv` — per-step IoU/loss/spill/piece traces, all runs
- `collected/by_config_arm.tsv` — means over pairs
- `collected/views/` — final exported views, 14 PNGs (7 pairs x 2 views)
- `manifest.tsv` — job IDs, weights, output dirs

Per-run `history.csv` and the ~2400 render frames per run stay on Oscar under
each job directory.
