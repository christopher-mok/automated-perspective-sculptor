# Deletion-proxy A/B: net vs spill

`di_net_20260724` re-runs the `srd_importance` arm with the **net** damage proxy
and is measured against the same arm from the previous di batch
`di_20260724`, which used the **spill** proxy. Everything else is held fixed
(full SRD, no count constraint, unconstrained theta, area weights at ratio
0.03125, `--min-steps 1500 --early-stop`, one seed on the seven standard pairs).

- **spill** = area x spill_fraction — penalizes a piece by how much of its own
  silhouette lands in negative space.
- **net** = area x (spill_fraction - coverage_fraction) — subtracts the useful
  area the piece covers, so a piece is only pushed toward the deletion offer when
  its spill outweighs what it contributes.

## Per-pair (final metrics, srd_importance)

| pair            | IoU net | IoU spill |   dIoU  | Tvsky net | Tvsky spill |  dTvsky | spill net | spill spill | del n | del s | patches n | patches s |
|-----------------|--------:|----------:|--------:|----------:|------------:|--------:|----------:|------------:|------:|------:|----------:|----------:|
| sun_moon        |  0.8129 |    0.8293 | -0.0163 |    0.9591 |      0.9655 | -0.0064 |    0.0206 |      0.0159 |     1 |     2 |        19 |        18 |
| water_fire      |  0.9504 |    0.9529 | -0.0025 |    0.9879 |      0.9884 | -0.0004 |    0.0084 |      0.0082 |     2 |     2 |        18 |        18 |
| cat_face_bass   |  0.8902 |    0.8826 | +0.0076 |    0.9771 |      0.9720 | +0.0052 |    0.0125 |      0.0174 |     3 |     5 |        17 |        15 |
| dance_argument  |  0.8214 |    0.8245 | -0.0030 |    0.9544 |      0.9509 | +0.0036 |    0.0273 |      0.0322 |     3 |     1 |        17 |        19 |
| horse_circle    |  0.9243 |    0.9152 | +0.0091 |    0.9829 |      0.9844 | -0.0016 |    0.0108 |      0.0075 |     1 |     0 |        19 |        20 |
| acm_scf         |  0.6535 |    0.6572 | -0.0037 |    0.9173 |      0.9171 | +0.0002 |    0.0322 |      0.0338 |     1 |     3 |        19 |        17 |
| teapot_droplets |  0.7470 |    0.7537 | -0.0066 |    0.9197 |      0.9204 | -0.0006 |    0.0519 |      0.0528 |     2 |     4 |        18 |        16 |
| **mean**        |  0.8285 |    0.8308 | -0.0022 |    0.9569 |      0.9569 | -0.0000 |    0.0234 |      0.0240 |  1.86 |  2.43 |      18.1 |      17.6 |

Head-to-head: IoU net > spill on 2/7 pairs, Tversky on 3/7, spill lower on 4/7.

## Read

The net proxy is a wash. Mean IoU is 0.0022 lower with net (spill marginally
better), mean Tversky is identical to four decimals, and mean spill is a hair
lower with net (0.0234 vs 0.0240). No pair moves by more than ~0.016 IoU, which
is within single-seed noise. The one consistent behavioral difference is that
net offers fewer pieces for deletion (mean 1.86 vs 2.43 deletes) and therefore
keeps slightly more patches (18.1 vs 17.6) — expected, since subtracting the
coverage term shrinks each piece's damage score and softens the deletion
pressure. That restraint does not translate into a quality gain here.

Conclusion: on this single-seed, seven-pair A/B, swapping the spill proxy for
the net proxy neither helps nor hurts fit quality; it only makes deletion
slightly more conservative. No reason to prefer net over the simpler spill proxy
on this evidence.
