#!/usr/bin/env python3
"""Package all successful experiment runs into experiments/<run>/ folders.

Every run that produced a complete result file gets its own self-contained
folder holding:

    report.txt      label, experiment kind and configuration, SLURM job info
                    (id, sbatch script, accounting record), the full result
                    report, and a listing of the artifacts
    artifacts/      copy of everything the run wrote (csv, png, txt, ...)
    logs/           copy of the run's sbatch script and stdout/stderr

Usage:  python compile_experiments.py [--out experiments] [--clean]
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

# ---------------------------------------------------------------- descriptions

BATCH_NOTES = {
    "20260717_1238": "SRD ablation batch (2000 steps, default swept-volume res)",
    "20260717_1249": "SRD ablation batch, resubmission of 20260717_1238",
    "20260718_0322": "patch-size sweep canary (single job smoke test)",
    "20260718_0323": "SRD vs. baseline comparison batch (incl. annealing arm)",
    "20260718_0340": "patch-size sweep batch (5/10/20/30 patches)",
    "20260718_1733": "combined ablation + comparison + patch-size batch",
    "20260718_1833": "combined ablation + comparison + patch-size batch (latest)",
    "20260719_2351": "resubmission of the two jobs that failed in 20260718_1833",
    "fanfill_20260720_0241": "fan-fill basic-vs-SRD comparison (2000 steps, IoU sampled every 25)",
    "sweep_20260720_0247": "silhouette/negative-space weight-ratio sweep (0.25/0.5/0.8/1/2/4, sum held at 4.5)",
}

KIND_NOTES = {
    "ablation": (
        "SRD ablation. Runs run_ablation.py in either 'full' mode (all SRD "
        "machinery on) or 'ablation' mode (a component disabled) so the two "
        "arms can be compared on the same image pair."
    ),
    "comparison": (
        "Method comparison. Same image pair optimized with SRD enabled "
        "(srd arm) versus the plain baseline optimizer (basic arm), 3 trials "
        "each with seeds 0,1,2."
    ),
    "patchsize": (
        "Patch-size sweep. compare_srd.py runs a with-SRD and a without-SRD "
        "configuration at a fixed initial patch count, 3 trials each."
    ),
    "convergence": (
        "Convergence study. compare_convergence.py optimizes until a plateau "
        "or IoU target is hit (rather than a fixed step budget) and records "
        "the steps/seconds required, one job per (pair, method, seed)."
    ),
    "fanfill": (
        "Fan-fill comparison. compare_fanfill.py runs the same basic-vs-SRD "
        "arms as the comparison batches but triangulates patches with a "
        "vertex-0 fan and reports IoU sampled over the run, one job per "
        "(pair, method, seed)."
    ),
    "weights": (
        "Loss-weight sweep. compare_convergence.py run at several "
        "silhouette:negative-space weight ratios with their sum held fixed, "
        "one job per (pair, ratio, seed)."
    ),
    "benchmark": (
        "Legacy multi-pair benchmark run (results/<name>/benchmark.txt), one "
        "configuration swept across all image pairs for 3 trials."
    ),
}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return ""


# ------------------------------------------------------------------ slurm info

def index_logs():
    """jobname -> {jobid, out, err, sbatch} from ./logs."""
    by_name = {}
    if not LOGS.is_dir():
        return by_name
    for p in sorted(LOGS.iterdir()):
        m = re.match(r"^(.*)\.(\d+)\.(out|err)$", p.name)
        if m:
            name, jobid, ext = m.groups()
            e = by_name.setdefault(name, {})
            e["jobid"] = jobid
            e[ext] = p
        elif p.name.endswith(".sbatch"):
            by_name.setdefault(p.stem, {})["sbatch"] = p
    return by_name


_SACCT_CACHE = {}


def sacct(jobid):
    if jobid in _SACCT_CACHE:
        return _SACCT_CACHE[jobid]
    out = sh([
        "sacct", "-j", str(jobid), "-P", "--format",
        "JobID,JobName%60,State,ExitCode,Elapsed,Start,End,Partition,NodeList,ReqTRES%80,MaxRSS",
    ])
    _SACCT_CACHE[jobid] = out.strip()
    return _SACCT_CACHE[jobid]


def sbatch_summary(path):
    """Pull the #SBATCH directives and the python command out of a sbatch file."""
    if not path or not path.exists():
        return [], None
    directives, cmd = [], None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#SBATCH"):
            directives.append(line.replace("#SBATCH ", "").strip())
        elif line.strip().startswith("python "):
            cmd = line.strip()
    return directives, cmd


def tail(path, n=15):
    if not path or not path.exists():
        return None
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


# ------------------------------------------------------------------ discovery

class Run:
    def __init__(self, label, kind, batch, resdir, report_files, note="",
                 jobname=None, jobid=None, extra=None):
        self.label = label
        self.kind = kind
        self.batch = batch
        self.resdir = resdir
        self.report_files = report_files      # list of Path
        self.note = note
        self.jobname = jobname
        self.jobid = jobid
        self.extra = extra or {}


def is_complete_output(p):
    t = p.read_text(errors="replace")
    return "Summary:" in t and re.search(r"completed_trials=([1-9]\d*)", t) is not None


def discover():
    runs = []

    # --- results/slurm/<batch>/<jobname>/
    for batch_dir in sorted((RESULTS / "slurm").glob("*")):
        if not batch_dir.is_dir():
            continue
        batch = batch_dir.name
        for job_dir in sorted(batch_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            name = job_dir.name
            kind = name.split("_")[0]
            kind = kind if kind in ("ablation", "comparison", "patchsize") else "other"
            out = job_dir / "output.txt"
            summary = job_dir / "SUMMARY.txt"
            if out.exists() and is_complete_output(out):
                reports = [out]
            elif summary.exists() and "SUCCESS" in summary.read_text(errors="replace"):
                reports = [summary] + sorted(job_dir.glob("patches_*/comparison.txt"))
            else:
                continue
            runs.append(Run(
                label=f"{batch}/{name}", kind=kind, batch=batch, resdir=job_dir,
                report_files=reports, note=BATCH_NOTES.get(batch, ""),
                jobname=f"{batch}_{name}",
            ))

    # --- manifest-driven batches (srd_ablation, convergence_*, fanfill_*,
    #     weights/<sweep>) — the weight sweeps nest one level deeper
    manifests = sorted(RESULTS.glob("*/manifest.tsv")) + \
        sorted(RESULTS.glob("*/*/manifest.tsv"))
    for manifest in manifests:
        base = manifest.parent
        if base.name.startswith("convergence"):
            kind = "convergence"
        elif base.name.startswith("fanfill"):
            kind = "fanfill"
        elif base.parent.name == "weights":
            kind = "weights"
        else:
            kind = "ablation"
        with manifest.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                d = Path(row["output_dir"])
                if not d.is_dir():
                    d = base / row["job_name"]
                reports = [p for p in (d / "convergence.txt", d / "fanfill.txt",
                                       d / "output.txt") if p.exists()]
                reports = [p for p in reports
                           if p.name != "output.txt" or is_complete_output(p)]
                if not reports:
                    continue
                if reports[0].name in ("convergence.txt", "fanfill.txt") and \
                        "stop_reason=" not in reports[0].read_text(errors="replace"):
                    continue
                runs.append(Run(
                    label=f"{base.name}/{row['job_name']}", kind=kind,
                    batch=base.name, resdir=d, report_files=reports,
                    note=BATCH_NOTES.get(base.name, ""),
                    jobname=row["job_name"], jobid=row.get("job_id"),
                    extra={k: v for k, v in row.items()
                           if k not in ("job_id", "job_name", "output_dir")},
                ))

    # --- legacy benchmark.txt runs
    for bm in sorted(RESULTS.glob("*/benchmark.txt")):
        runs.append(Run(
            label=bm.parent.name.replace(" ", "_"), kind="benchmark",
            batch="legacy", resdir=bm.parent, report_files=[bm],
            note="Legacy pre-SLURM run, executed interactively; no job record.",
        ))

    # --- standalone srd comparisons
    for comp in sorted(RESULTS.glob("srd_comparison_*/output.txt")):
        if is_complete_output(comp):
            runs.append(Run(
                label=comp.parent.name, kind="comparison", batch="legacy",
                resdir=comp.parent, report_files=[comp],
                note="Standalone SRD comparison run; no job record.",
            ))
    return runs


# -------------------------------------------------------------------- emitting

def rel(p):
    return p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p


def artifacts(d):
    rows = []
    for p in sorted(d.rglob("*")):
        if p.is_file():
            rows.append(f"  {p.relative_to(d)}  ({p.stat().st_size} bytes)")
    return rows


def copy_tree(src, dst):
    """Copy src into dst, returning the number of files copied."""
    n = 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        target = dst / p.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        n += 1
    return n


def write_report(run, log_index, out_dir):
    pkg = out_dir / run.label.replace("/", "__")
    pkg.mkdir(parents=True, exist_ok=True)

    L = []
    bar = "=" * 78
    L += [bar, f"EXPERIMENT: {run.label}", bar, ""]
    L += [f"kind          : {run.kind}",
          f"batch         : {run.batch}",
          f"results_dir   : {rel(run.resdir)}"]
    for k, v in run.extra.items():
        L.append(f"{k:<14}: {v}")
    L += ["", "DESCRIPTION", "-" * 78]
    L += ["  " + KIND_NOTES.get(run.kind, "Unclassified run.")]
    if run.note:
        L += ["", f"  Batch note: {run.note}"]

    # slurm
    info = log_index.get(run.jobname, {}) if run.jobname else {}
    jobid = run.jobid or info.get("jobid")
    L += ["", "SLURM JOB", "-" * 78]
    if jobid:
        L.append(f"  job_id   : {jobid}")
        L.append(f"  job_name : {run.jobname}")
        for p in (info.get("sbatch"), info.get("out"), info.get("err")):
            if p:
                L.append(f"  log      : {p.relative_to(ROOT)}")
        acct = sacct(jobid)
        if acct:
            L += ["", "  sacct:"] + [f"    {l}" for l in acct.splitlines()]
    else:
        L.append("  (no SLURM record — run interactively or logs unavailable)")

    directives, cmd = sbatch_summary(info.get("sbatch"))
    if directives:
        L += ["", "  sbatch directives:"] + [f"    {d}" for d in directives]
    if cmd:
        L += ["", "  command:", f"    {cmd}"]

    t = tail(info.get("err"), 15)
    if t and t.strip():
        L += ["", "  stderr (last 15 lines):"] + [f"    {l}" for l in t.splitlines()]

    # reports
    for rf in run.report_files:
        L += ["", bar, f"RESULT FILE: {rf.relative_to(run.resdir)}", bar, ""]
        L.append(rf.read_text(errors="replace").rstrip())

    L += ["", bar, "ARTIFACTS  (copied into ./artifacts/)", bar]
    L += [f"  source: {rel(run.resdir)}"]
    L += artifacts(run.resdir)
    L.append("")

    n_art = copy_tree(run.resdir, pkg / "artifacts")

    n_log = 0
    for p in (info.get("sbatch"), info.get("out"), info.get("err")):
        if p and p.exists():
            (pkg / "logs").mkdir(exist_ok=True)
            shutil.copy2(p, pkg / "logs" / p.name)
            n_log += 1

    (pkg / "report.txt").write_text("\n".join(L))
    return pkg.name, n_art, n_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--clean", action="store_true",
                    help="remove existing contents of the output dir first")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    if args.clean and out_dir.is_dir():
        for p in sorted(out_dir.iterdir()):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    out_dir.mkdir(exist_ok=True)

    log_index = index_logs()
    runs = discover()

    index = ["Packaged experiment runs — one folder per run",
             "  <run>/report.txt   config, SLURM record and full result report",
             "  <run>/artifacts/   the run's own output (csv, png, txt, ...)",
             "  <run>/logs/        sbatch script and stdout/stderr",
             "",
             f"source repo : {ROOT}",
             f"experiments : {len(runs)} successful runs", ""]
    by_kind = {}
    for r in runs:
        by_kind.setdefault(r.kind, []).append(r)

    total_art = 0
    for kind in sorted(by_kind):
        index += [f"[{kind}]  {KIND_NOTES.get(kind, '').split('.')[0]}"]
        for r in sorted(by_kind[kind], key=lambda r: r.label):
            name, n_art, n_log = write_report(r, log_index, out_dir)
            total_art += n_art
            index.append(f"  {name + '/':<62}  {r.batch}"
                         f"  ({n_art} artifacts, {n_log} logs)")
        index.append("")

    (out_dir / "INDEX.txt").write_text("\n".join(index))
    print(f"packaged {len(runs)} runs ({total_art} artifact files) into {out_dir}")


if __name__ == "__main__":
    main()
