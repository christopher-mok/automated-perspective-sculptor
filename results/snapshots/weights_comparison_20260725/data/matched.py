"""Weight comparison restricted to the lambda-count=0 family, on the set of
pairs every one of those sweeps has actually finished.
"""
import csv, glob, os, re
from collections import defaultdict

HINGE = "/oscar/home/cjmok/aps-hinge2"
ORIG = "/oscar/home/cjmok/aps-original"

LAM0 = {
    "neg_minarea_nonorm4_20260724":      ("nonorm4",    "srd_restart", 0.005),
    "neg_minarea_aw0p1875_lam0_20260725": ("aw0p1875",  "srd_restart", 0.005),
    "neg_awrec_srd_20260725":            ("aw0p073",    "srd",         0.005),
    "neg_restart_aw0p25_lam0_20260724":  ("aw0p25",     "srd_restart", 0.010),
}

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

data = defaultdict(dict)   # sweep -> pair -> row
for sweep, (cfg, arm, mpa) in LAM0.items():
    p = f"{HINGE}/results/negspace/{sweep}/collected/summary.tsv"
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p), delimiter="\t"):
        if fnum(r.get("final_mean_iou")) is None:
            continue
        data[sweep][r["pair"]] = r

# original branch
for rep in sorted(glob.glob(f"{ORIG}/results/original/orig_20260724/*/report.txt")):
    text = open(rep).read()
    def grab(k):
        m = re.search(rf"^\s*{k}=(.+)$", text, re.M)
        return m.group(1).strip() if m else None
    pair = os.path.basename(os.path.dirname(rep)).replace("orig_", "")
    data["ORIGINAL_orig_20260724"][pair] = {
        "final_mean_iou": grab("final_mean_iou"),
        "final_mean_spill": grab("final_mean_spill"),
        "final_mean_coverage": grab("final_mean_coverage"),
        "final_mean_precision": grab("final_mean_precision"),
        "final_patches": grab("final_patches"),
        "final_step": grab("final_step"),
    }
LABEL = dict(LAM0)
LABEL["ORIGINAL_orig_20260724"] = ("nonorm1p25", "srd_restart[ORIG branch]", None)

sets = {s: set(d) for s, d in data.items()}
for s, ps in sorted(sets.items()):
    print(f"{s:<38} {len(ps):>2} done: {' '.join(sorted(ps))}")
common = set.intersection(*sets.values()) if sets else set()
print()
print(f"COMMON to all: {sorted(common)}  (n={len(common)})")
print()

def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")

print("=" * 96)
print(f"MATCHED on the {len(common)} pairs every lambda-count=0 sweep finished")
print("=" * 96)
print(f"{'config':<11} {'arm':<26} {'minarea':>8} {'IoU':>8} {'spill':>8} "
      f"{'cov':>7} {'prec':>7} {'pieces':>7}")
print("-" * 84)
rank = []
for sweep in sorted(data, key=lambda s: -mean(fnum(data[s][p]["final_mean_iou"]) for p in common)):
    cfg, arm, mpa = LABEL[sweep]
    rows = [data[sweep][p] for p in common]
    iou = mean(fnum(r["final_mean_iou"]) for r in rows)
    rank.append((cfg, iou))
    print(f"{cfg:<11} {arm:<26} {str(mpa or '-'):>8} {iou:>8.4f} "
          f"{mean(fnum(r['final_mean_spill']) for r in rows):>8.4f} "
          f"{mean(fnum(r['final_mean_coverage']) for r in rows):>7.3f} "
          f"{mean(fnum(r['final_mean_precision']) for r in rows):>7.3f} "
          f"{mean(fnum(r['final_patches']) for r in rows):>7.1f}")

print()
print("=" * 96)
print("PER-PAIR final IoU on the common set")
print("=" * 96)
hdr = sorted(data, key=lambda s: LABEL[s][0])
print(f"{'pair':<16}" + "".join(f"{LABEL[s][0]:>13}" for s in hdr))
print("-" * (16 + 13 * len(hdr)))
for p in sorted(common):
    line = f"{p:<16}"
    vals = {s: fnum(data[s][p]["final_mean_iou"]) for s in hdr}
    best = max(vals.values())
    for s in hdr:
        v = vals[s]
        line += f"{('*' if v == best else ' ') + f'{v:.4f}':>13}"
    print(line)
print()
print(f"{'pair':<16}" + "".join(f"{LABEL[s][0]:>13}" for s in hdr) + "   (spill)")
print("-" * (16 + 13 * len(hdr)))
for p in sorted(common):
    line = f"{p:<16}"
    vals = {s: fnum(data[s][p]["final_mean_spill"]) for s in hdr}
    best = min(vals.values())
    for s in hdr:
        v = vals[s]
        line += f"{('*' if v == best else ' ') + f'{v:.4f}':>13}"
    print(line)
