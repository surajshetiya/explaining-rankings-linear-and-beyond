#!/usr/bin/env python3
"""aggregate_kt_results.py -- collect Kendall-Tau errors written by run_kt_experiments.sh.

Usage:  python3 aggregate_kt_results.py OUT_DIR

Reads OUT_DIR/manifest.tsv and OUT_DIR/json/*.json and writes
OUT_DIR/results.csv (one row per run) and OUT_DIR/summary_by_config.csv
(per configuration, over seeds), then prints a leaderboard.

Environment (optional): REF_OPT_LINEAR, REF_OPT_POLY = known optimal KT for the
linear / polynomial family. Otherwise the best KT found is used as reference.
"""
import csv, json, math, os, statistics, sys
from collections import defaultdict

out = sys.argv[1]
env_ref = {"linear": os.environ.get("REF_OPT_LINEAR", "").strip(),
           "poly": os.environ.get("REF_OPT_POLY", "").strip()}

latest = {}
with open(os.path.join(out, "manifest.tsv")) as f:
    for m in csv.DictReader(f, delimiter="\t"):
        latest[m["id"]] = m          # later lines (re-runs) win

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def group_of(r):
    return "linear" if r["family"] == "linear" else "poly_d" + r["degree"]

rows = []
for m in latest.values():
    r = dict(m)
    r["group"] = group_of(m)
    path = os.path.join(out, "json", m["id"] + ".json")
    d = {}
    if m["status"] in ("ok", "cached") and os.path.exists(path):
        try:
            d = json.load(open(path))
        except Exception:
            r["status"] = "bad_json"
    kt = fnum(d.get("kendall_tau"))
    r["kt_error"] = kt
    r["kendall_agreement"] = d.get("kendall_agreement")
    r["total_comparisons"] = d.get("total_comparisons")
    r["forced_tie_disagreements"] = d.get("forced_tie_disagreements")
    r["optimizable_comparisons"] = d.get("optimizable_comparisons")
    r["runtime_s"] = d.get("total_runtime_seconds")
    r["approximation_factor"] = d.get("approximation_factor")
    r["sampling_bound_met"] = d.get("sampling_bound_met")
    r["guaranteed_fraction"] = d.get("guaranteed_fraction_of_optimum")
    # Certified lower bound on OPT_KT from the closest-ranking IIS packing:
    # deleted_count <= K * OPT_KT  =>  OPT_KT >= ceil(deleted_count / K)
    r["cert_opt_lower_bound"] = None
    if d.get("deleted_comparisons") is not None and d.get("approximation_factor"):
        deleted = len(d["deleted_comparisons"]) + (d.get("forced_tie_disagreements") or 0)
        r["cert_opt_lower_bound"] = math.ceil(deleted / d["approximation_factor"] - 1e-12)
    rows.append(r)

ok = [r for r in rows if r["kt_error"] is not None]
best_known = {}
for r in ok:
    g = r["group"]
    best_known[g] = min(best_known.get(g, math.inf), r["kt_error"])
global_best = min((r["kt_error"] for r in ok), default=None)
lower_bound = {}
for r in ok:
    if r["cert_opt_lower_bound"] is not None:
        g = r["group"]
        lower_bound[g] = max(lower_bound.get(g, 0), r["cert_opt_lower_bound"])

def ref_for(g):
    key = "linear" if g == "linear" else "poly"
    v = fnum(env_ref[key])
    if v is not None:
        return v, "user_optimum"
    return best_known.get(g), "best_known"

def ratio(kt, ref):
    if kt is None or ref is None:
        return None
    if ref == 0:
        return 1.0 if kt == 0 else math.inf
    return kt / ref

for r in rows:
    ref, kind = ref_for(r["group"])
    r["reference_kt"], r["reference_kind"] = ref, kind
    r["kt_over_ref"] = ratio(r["kt_error"], ref)
    r["gap_to_ref"] = None if r["kt_error"] is None or ref is None else r["kt_error"] - ref
    r["gap_to_global_best"] = None if r["kt_error"] is None else r["kt_error"] - global_best
    r["group_cert_lower_bound"] = lower_bound.get(r["group"])
    tc = fnum(r["total_comparisons"])
    r["kt_normalized"] = None if (r["kt_error"] is None or not tc) else r["kt_error"] / tc

cols = ["group", "family", "method", "degree", "params", "seed", "status", "kt_error",
        "kt_normalized", "reference_kt", "reference_kind", "kt_over_ref", "gap_to_ref",
        "gap_to_global_best", "group_cert_lower_bound", "cert_opt_lower_bound",
        "kendall_agreement", "total_comparisons", "forced_tie_disagreements",
        "optimizable_comparisons", "approximation_factor", "sampling_bound_met",
        "guaranteed_fraction", "runtime_s", "wall_seconds", "id"]
rows.sort(key=lambda r: (r["group"], r["kt_error"] if r["kt_error"] is not None else math.inf, r["id"]))
with open(os.path.join(out, "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# Per-configuration summary across seeds
cfg = defaultdict(list)
for r in rows:
    cfg[(r["group"], r["method"], r["params"])].append(r)
scols = ["group", "method", "params", "runs", "ok_runs", "kt_min", "kt_mean", "kt_std",
         "kt_max", "reference_kt", "reference_kind", "ratio_min", "ratio_mean", "mean_runtime_s"]
srows = []
for (g, m, p), rs in cfg.items():
    kts = [r["kt_error"] for r in rs if r["kt_error"] is not None]
    rts = [fnum(r["runtime_s"]) for r in rs if fnum(r["runtime_s"]) is not None]
    ref, kind = ref_for(g)
    srows.append({
        "group": g, "method": m, "params": p, "runs": len(rs), "ok_runs": len(kts),
        "kt_min": min(kts) if kts else None,
        "kt_mean": statistics.mean(kts) if kts else None,
        "kt_std": statistics.pstdev(kts) if len(kts) > 1 else (0.0 if kts else None),
        "kt_max": max(kts) if kts else None,
        "reference_kt": ref, "reference_kind": kind,
        "ratio_min": ratio(min(kts), ref) if kts else None,
        "ratio_mean": ratio(statistics.mean(kts), ref) if kts else None,
        "mean_runtime_s": statistics.mean(rts) if rts else None})
srows.sort(key=lambda r: (r["group"], r["kt_min"] if r["kt_min"] is not None else math.inf))
with open(os.path.join(out, "summary_by_config.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=scols)
    w.writeheader()
    w.writerows(srows)

# Console leaderboard: best config per (group, method)
print("\n=== Best KT error per method (reference = %s) ===" % (
    "user-supplied optimum where given, else best known"))
best = {}
for s in srows:
    if s["kt_min"] is None:
        continue
    k = (s["group"], s["method"])
    if k not in best or s["kt_min"] < best[k]["kt_min"]:
        best[k] = s
cur = None
for (g, m), s in sorted(best.items(), key=lambda kv: (kv[0][0], kv[1]["kt_min"])):
    if g != cur:
        ref, kind = ref_for(g)
        lb = lower_bound.get(g)
        print("\n[%s] reference KT = %s (%s)%s" % (
            g, ref, kind, "; certified OPT lower bound = %s" % lb if lb is not None else ""))
        print("  %-26s %8s %8s %8s   %s" % ("method", "KT min", "KT mean", "ratio", "config"))
        cur = g
    print("  %-26s %8s %8.1f %8s   %s" % (
        m, int(s["kt_min"]), s["kt_mean"],
        "%.3f" % s["ratio_min"] if s["ratio_min"] not in (None, math.inf) else "n/a", s["params"]))
bad = [r for r in rows if r["kt_error"] is None]
print("\n%d runs have no KT result (see manifest.tsv / logs/)." % len(bad))
print("Wrote %s and %s" % (os.path.join(out, "results.csv"), os.path.join(out, "summary_by_config.csv")))
