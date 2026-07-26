"""Cross-sweep weight comparison for the SRD-family arms, plus an
original-branch vs main-branch check at matched weights.
"""
import csv
import glob
import os
import re
from collections import defaultdict

HINGE = "/oscar/home/cjmok/aps-hinge2"
ORIG = "/oscar/home/cjmok/aps-original"

# The weight configs, as (silhouette, negative_space, normalized), and the
# equivalent normalized silhouette:negative-space ratio at f_fg = 0.153
# (the water_fire + sun_moon mean) so every config lands on one axis.
CONFIG_META = {
    "nonorm1p25": (2.0, 2.5, False),
    "nonorm4":    (0.9, 3.6, False),
    "aw0p073":    (0.306, 4.194, True),
    "aw0p125":    (0.5, 4.0, True),
    "aw0p1875":   (0.7105, 3.7895, True),
    "aw0p25":     (0.9, 3.6, True),
}
F_REF = 0.153


def effective_ratio(tag):
    """Normalized-equivalent silhouette:negative-space ratio."""
    if tag not in CONFIG_META:
        return None
    s, n, normalized = CONFIG_META[tag]
    if normalized:
        return s / n
    return (s * F_REF) / (s * (1 - F_REF) + n)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


rows = []
for path in sorted(glob.glob(f"{HINGE}/results/negspace/*/collected/summary.tsv")):
    sweep = path.split("/results/negspace/")[1].split("/")[0]
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if fnum(r.get("final_mean_iou")) is None:
                continue
            rows.append({
                "sweep": sweep, "branch": "main",
                "pair": r["pair"], "arm": r["arm"], "config": r["config"],
                "iou": fnum(r["final_mean_iou"]),
                "best_iou": fnum(r.get("best_mean_iou")),
                "spill": fnum(r.get("final_mean_spill")),
                "cov": fnum(r.get("final_mean_coverage")),
                "prec": fnum(r.get("final_mean_precision")),
                "patches": fnum(r.get("final_patches")),
                "step": fnum(r.get("final_step")),
                "secs": fnum(r.get("total_seconds")),
                "stop": r.get("stop_reason", "-"),
            })

# original branch: no collector, parse report.txt directly
for rep in sorted(glob.glob(f"{ORIG}/results/original/orig_20260724/*/report.txt")):
    text = open(rep).read()
    def grab(k):
        m = re.search(rf"^\s*{k}=(.+)$", text, re.M)
        return fnum(m.group(1)) if m else None
    stop = re.search(r"^\s*stop_reason=(.+)$", text, re.M)
    rows.append({
        "sweep": "orig_20260724", "branch": "ORIGINAL",
        "pair": os.path.basename(os.path.dirname(rep)).replace("orig_", ""),
        "arm": "srd_restart", "config": "nonorm1p25",
        "iou": grab("final_mean_iou"), "best_iou": grab("best_mean_iou"),
        "spill": grab("final_mean_spill"), "cov": grab("final_mean_coverage"),
        "prec": grab("final_mean_precision"), "patches": grab("final_patches"),
        "step": grab("final_step"), "secs": grab("total_seconds"),
        "stop": stop.group(1).strip() if stop else "-",
    })

SRD_ARMS = {"srd", "srd_restart", "importnet"}
srd = [r for r in rows if r["arm"] in SRD_ARMS]

print("=" * 100)
print("SRD-FAMILY RUNS BY WEIGHT CONFIG  (no count objective)")
print("=" * 100)
print(f"{'config':<11} {'eff.ratio':>9} {'sweep':<36} {'arm':<12} {'n':>3} "
      f"{'IoU':>7} {'spill':>7} {'cov':>6} {'prec':>6} {'pieces':>7}")
print("-" * 100)

groups = defaultdict(list)
for r in srd:
    groups[(r["config"], r["sweep"], r["arm"], r["branch"])].append(r)


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


ordered = sorted(groups.items(),
                 key=lambda kv: (effective_ratio(kv[0][0]) or 9, kv[0][1]))
for (cfg, sweep, arm, branch), rs in ordered:
    er = effective_ratio(cfg)
    tag = sweep if branch == "main" else sweep + " [ORIGINAL]"
    print(f"{cfg:<11} {er if er else 0:>9.4f} {tag:<36} {arm:<12} {len(rs):>3} "
          f"{mean(r['iou'] for r in rs):>7.4f} {mean(r['spill'] for r in rs):>7.4f} "
          f"{mean(r['cov'] for r in rs):>6.3f} {mean(r['prec'] for r in rs):>6.3f} "
          f"{mean(r['patches'] for r in rs):>7.1f}")

print()
print("=" * 100)
print("POOLED BY WEIGHT CONFIG (all SRD-family arms and sweeps together)")
print("=" * 100)
bycfg = defaultdict(list)
for r in srd:
    bycfg[r["config"]].append(r)
print(f"{'config':<11} {'eff.ratio':>9} {'n':>4} {'IoU':>8} {'best':>8} "
      f"{'spill':>8} {'cov':>7} {'prec':>7} {'pieces':>7}")
print("-" * 76)
for cfg, rs in sorted(bycfg.items(), key=lambda kv: effective_ratio(kv[0]) or 9):
    er = effective_ratio(cfg)
    print(f"{cfg:<11} {er if er else 0:>9.4f} {len(rs):>4} "
          f"{mean(r['iou'] for r in rs):>8.4f} {mean(r['best_iou'] for r in rs):>8.4f} "
          f"{mean(r['spill'] for r in rs):>8.4f} {mean(r['cov'] for r in rs):>7.3f} "
          f"{mean(r['prec'] for r in rs):>7.3f} {mean(r['patches'] for r in rs):>7.1f}")

print()
print("=" * 100)
print("ORIGINAL BRANCH vs MAIN BRANCH, matched weights (nonorm1p25 = 2.0/2.5)")
print("=" * 100)
matched = [r for r in rows if r["config"] == "nonorm1p25" and r["arm"] in SRD_ARMS]
bykey = defaultdict(dict)
for r in matched:
    key = r["pair"]
    label = "ORIGINAL" if r["branch"] == "ORIGINAL" else f"main/{r['sweep'][:28]}"
    bykey[key][label] = r
print(f"{'pair':<16} {'source':<34} {'arm':<12} {'IoU':>8} {'spill':>8} "
      f"{'pieces':>7} {'step':>6} {'stop':>12}")
print("-" * 100)
for pair in sorted(bykey):
    for label, r in sorted(bykey[pair].items()):
        print(f"{pair:<16} {label:<34} {r['arm']:<12} {r['iou']:>8.4f} "
              f"{r['spill']:>8.4f} {r['patches']:>7.0f} {r['step']:>6.0f} {r['stop']:>12}")
    print()

print()
print("=" * 100)
print("PER-PAIR, best weight config by final IoU (SRD family)")
print("=" * 100)
bypair = defaultdict(list)
for r in srd:
    bypair[r["pair"]].append(r)
print(f"{'pair':<16} {'winner cfg':<11} {'IoU':>8} {'spill':>8} {'pieces':>7}   "
      f"{'runner-up':<11} {'IoU':>8}")
print("-" * 84)
wins = defaultdict(int)
for pair, rs in sorted(bypair.items()):
    rs = sorted([r for r in rs if r["iou"] is not None],
                key=lambda r: -r["iou"])
    if not rs:
        continue
    w = rs[0]
    wins[w["config"]] += 1
    ru = next((r for r in rs[1:] if r["config"] != w["config"]), None)
    print(f"{pair:<16} {w['config']:<11} {w['iou']:>8.4f} {w['spill']:>8.4f} "
          f"{w['patches']:>7.0f}   {ru['config'] if ru else '-':<11} "
          f"{ru['iou'] if ru else 0:>8.4f}")
print()
print("config win counts:", dict(sorted(wins.items(), key=lambda kv: -kv[1])))
