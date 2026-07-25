# Weight comparison and swept-volume ablation — 2026-07-25 (complete)

Supersedes the partial version in
`results/snapshots/weights_comparison_20260725/`, which was cut when only 3
pairs were common to every sweep. Six are now common and the ORIGINAL branch
batch is complete at 7/7.

## Reminder: lambda dominates weights

Sweeps at `--lambda-count 0.05` (the default) settle at ~17 pieces and IoU
0.77–0.85; at `--lambda-count 0` they reach ~29 pieces and 0.89–0.91. That
5–10 point effect is larger than the entire weight range measured below.
Everything here is at `--lambda-count 0`.

## Weight comparison

SRD family, `--lambda-count 0`, matched on the 6 pairs every sweep finished:
acm_scf, cat_face_bass, horse_circle, sun_moon, teapot_droplets, water_fire.
"ratio" is the normalized-equivalent silhouette:negative-space ratio at
f_fg = 0.153, putting normalized and non-normalized configs on one axis.

| config | ratio | arm | IoU | pieces | spill | coverage | precision |
|---|---|---|---|---|---|---|---|
| **aw0p1875** | 0.1875 | srd_restart | **0.9127** | 31.8 | 0.0655 | 0.967 | 0.942 |
| aw0p25 | 0.2500 | srd_restart | 0.9067 | 27.5 | 0.0788 | **0.971** | 0.931 |
| nonorm1p25 *(ORIGINAL)* | 0.0730 | srd_restart | 0.9042 | 28.3 | 0.0594 | 0.953 | 0.946 |
| aw0p073 | 0.0730 | srd *(no restart)* | 0.9022 | 30.8 | 0.0382 | 0.936 | 0.963 |
| nonorm4 | 0.0316 | srd_restart | 0.8900 | 29.5 | **0.0221** | 0.909 | **0.977** |

**Spill orders perfectly by effective ratio** — 0.0221, 0.0382, 0.0594, 0.0655,
0.0788 against ratios 0.032, 0.073, 0.073, 0.188, 0.250. Not one inversion, and
`nonorm4` wins spill on all six pairs individually. Precision and coverage order
the same way, in opposite directions. The pixel-count model predicts this axis
exactly.

**IoU spans only 2.3 points across the entire weight range** (0.890–0.913),
against lambda's 5–10. Weights buy you the spill/coverage tradeoff; they are not
where the IoU comes from.

### Per-pair final IoU

| pair | nonorm4 | aw0p073 | nonorm1p25 | aw0p1875 | aw0p25 |
|---|---|---|---|---|---|
| acm_scf | 0.8323 | 0.8376 | 0.8029 | 0.8662 | **0.8663** |
| cat_face_bass | 0.9452 | 0.9477 | 0.9501 | **0.9504** | 0.9392 |
| horse_circle | **0.9321** | 0.8993 | 0.9067 | 0.8948 | 0.8943 |
| sun_moon | 0.9225 | 0.9320 | 0.9511 | **0.9685** | 0.9496 |
| teapot_droplets | 0.7718 | **0.8495** | 0.8456 | 0.8325 | 0.8177 |
| water_fire | 0.9360 | 0.9472 | 0.9685 | 0.9636 | **0.9728** |
| **mean** | 0.8900 | 0.9022 | 0.9042 | **0.9127** | 0.9067 |

No config wins more than two pairs. horse_circle prefers the most
negative-space-weighted config and teapot the middle; acm and water_fire prefer
the most silhouette-weighted. The per-pair optimum genuinely varies.

### Per-pair final piece count

| pair | nonorm4 | aw0p073 | nonorm1p25 | aw0p1875 | aw0p25 |
|---|---|---|---|---|---|
| acm_scf | 34 | 37 | 31 | 46 | 30 |
| cat_face_bass | 29 | 31 | 25 | 30 | 25 |
| horse_circle | 32 | 35 | 27 | 32 | 29 |
| sun_moon | 27 | 24 | 30 | 26 | 26 |
| teapot_droplets | 33 | 32 | 33 | 33 | 30 |
| water_fire | 22 | 26 | 24 | 24 | 25 |
| **mean** | 29.5 | 30.8 | 28.3 | 31.8 | 27.5 |

Piece count shows **no relationship to weights** — the 27.5–31.8 spread has no
ordering with ratio. All start at 20, so every config grows by roughly half
again. Piece count is set by lambda and the SRD acceptance test, not by weights.

### Per-pair final spill

| pair | nonorm4 | aw0p073 | nonorm1p25 | aw0p1875 | aw0p25 |
|---|---|---|---|---|---|
| acm_scf | **0.0488** | 0.0646 | 0.1269 | 0.1292 | 0.1207 |
| cat_face_bass | **0.0103** | 0.0144 | 0.0229 | 0.0288 | 0.0448 |
| horse_circle | **0.0085** | 0.0196 | 0.0656 | 0.0407 | 0.0699 |
| sun_moon | **0.0087** | 0.0268 | 0.0273 | 0.0152 | 0.0357 |
| teapot_droplets | **0.0501** | 0.0791 | 0.1036 | 0.1569 | 0.1807 |
| water_fire | **0.0060** | 0.0245 | 0.0099 | 0.0220 | 0.0210 |
| **mean** | 0.0221 | 0.0382 | 0.0594 | 0.0655 | 0.0788 |

### On aw0p073

Still 4th on IoU, but with six pairs it is 0.9022 against ORIGINAL's 0.9042 — a
**0.2 point gap**, down from 1.4 on the three-pair set. And it does that at the
second-lowest spill (0.0382 vs 0.0594) and second-highest precision. Given it is
still the only arm here without conflict restart, the derived weights look
essentially equivalent to the original weights they were derived to match, with
better edges. `neg_awrec_restart_20260725` (same config, restart added) is
running and will settle this.

## Swept-volume ablation

`nonorm1p25`, 32 SRD candidates, `--lambda-count 0`. Matched on the 3 pairs all
three arms finished — the swept-adds-ON arm is only 3/7 so far, so this is
provisional.

| swept adds | restart | IoU | pieces | spill | coverage | precision |
|---|---|---|---|---|---|---|
| **ON** | no | **0.8873** | **35.0** | 0.0544 | 0.932 | 0.949 |
| OFF | no | 0.7874 | 20.0 | 0.0837 | 0.852 | 0.921 |
| OFF | yes | 0.8410 | 20.3 | 0.0864 | 0.908 | 0.920 |

| pair | swept ON / no restart | swept OFF / no restart | swept OFF / restart |
|---|---|---|---|
| acm_scf | 0.8257 | 0.6636 | 0.7494 |
| cat_face_bass | 0.9435 | 0.8879 | 0.9105 |
| sun_moon | 0.8927 | 0.8108 | 0.8632 |

**Disabling swept-volume-guided additions costs 10 IoU points** (0.887 → 0.787),
consistent on all three pairs and largest on acm_scf at −16. Restart recovers
about half of it (+5.4), but does not close the gap.

**The piece count is the striking part.** Without swept-volume guidance, runs end
at exactly **20.0 pieces** — their starting count — and 20.3 with restart. With
guidance they reach 35. Additions placed without swept-volume guidance are
essentially never accepted by SRD's acceptance test. The swept volume is not
merely improving where new pieces go; it is the reason any of them survive at
all.

Caveat: 32 candidates here against 64 in the weight sweeps, so these arms are not
directly comparable to the table above — only to each other.

## Data and images

Per sweep, under `<sweep>/collected/`:

- `summary.tsv` — 27 metrics per run including `final_mean_iou`,
  `final_patches`, `best_mean_iou`, spill/precision/coverage, SRD add/split/
  delete totals, `stop_reason`
- `curves.csv` — per-step traces for every run in the sweep
- `by_config_arm.tsv` — means over pairs
- `views/` — **final exported render for every run, both views**
- `manifest.tsv` at the sweep root — job IDs, weights, output dirs

Sweeps included: `neg_minarea_nonorm4_20260724`,
`neg_minarea_aw0p1875_lam0_20260725`, `neg_awrec_srd_20260725`,
`neg_restart_aw0p25_lam0_20260724`, `neg_swept_norestart_nonorm1p25_20260725`,
`neg_noswept_nonorm1p25_20260725`, `neg_noswept_restart_nonorm1p25_20260725`,
`neg_h2ri_nonorm1p25_20260725`.

The ORIGINAL-branch batch lives in the `aps-original` checkout on branch
`original-baseline` and is not duplicated here.

Per-run `history.csv` and the ~2400 render frames per run stay on Oscar.

## Still running

- `neg_awrec_srd_20260725` — 7/8 (dance_argument outstanding)
- `neg_awrec_restart_20260725` — 0/7, the restart partner for aw0p073
- `neg_h2r_aw0p1875_20260725` — 0/7, hinge2+restart at aw0p1875 without
  importance deletion
- `neg_swept_norestart_nonorm1p25_20260725` — 3/7, the swept-ON ablation arm

Single seed (0) throughout. No error bars.
