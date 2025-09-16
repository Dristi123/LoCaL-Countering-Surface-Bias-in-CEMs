#!/usr/bin/env python3
import json, math
from pathlib import Path

INPUT = Path("combined_custom_sample.jsonl")
SEM, SURF = "score", "surfaceSim"
METRICS = ["codebleu", "codebertscore", "crystalbleu", "codescore"]

STEP = 0.05
MIN_DFS, MIN_SFD, MIN_CTRL = 50, 50, 150

def f2(x):
    try: return float(x)
    except: return float("nan")

# ---- MAE instead of MSE ----
def mae(pairs):
    xs = [abs(s - m) for s, m in pairs if not (math.isnan(s) or math.isnan(m))]
    return (sum(xs)/len(xs)) if xs else math.nan
# ----------------------------

def vals(lo, hi, step):
    n = int(round((hi - lo)/step))
    return [round(lo + i*step, 2) for i in range(n+1)]

def infer_source(obj):
    rid = str(obj.get("id", ""))
    if rid.startswith("SHARE"): return "sharecode"
    if rid.startswith("LOCAL"): return "local"
    return "baseline"

TX_LOS = vals(0.10, 0.65, STEP)
TX_HIS = vals(0.40, 0.90, STEP)
TY_LOS = vals(0.10, 0.60, STEP)
TY_HIS = vals(0.40, 0.90, STEP)

rows = []
with INPUT.open("r", encoding="utf-8") as f:
    for ln in f:
        s = ln.strip()
        if not s: continue
        obj = json.loads(s)
        y = f2(obj.get(SEM)); x = f2(obj.get(SURF))
        if math.isnan(x) or math.isnan(y): continue
        obj["_x"] = x; obj["_y"] = y; obj["_src"] = infer_source(obj)
        for k in METRICS:
            if k in obj: obj[k] = f2(obj[k])
        rows.append(obj)

metrics = [m for m in METRICS if any(not math.isnan(r.get(m, math.nan)) for r in rows)]

results = []  # (tx_lo, tx_hi, ty_lo, ty_hi, obj, used, n_dfs, n_sfd, n_ctrl, per_metric, by_src)

for tx_lo in TX_LOS:
    for tx_hi in TX_HIS:
        for ty_lo in TY_LOS:
            for ty_hi in TY_HIS:
                if tx_hi <= tx_lo or ty_hi <= ty_lo:
                    continue
                dfs = {m: [] for m in metrics}
                sfd = {m: [] for m in metrics}
                ctrl = {m: [] for m in metrics}
                n_dfs = n_sfd = n_ctrl = 0
                by_src = {}

                for r in rows:
                    x = r["_x"]; y = r["_y"]; src = r["_src"]

                    # DFS (top-left corner)
                    if (x <= tx_lo) and (y >= ty_hi):
                        n_dfs += 1; bucket = dfs; reg = "DFS"
                    # SFD (bottom-right corner)
                    elif (x >= tx_hi) and (y <= ty_lo):
                        n_sfd += 1; bucket = sfd; reg = "SFD"
                    # CONTROL = everything else
                    else:
                        n_ctrl += 1; bucket = ctrl; reg = "Control"

                    for m in metrics:
                        bucket[m].append((y, r.get(m, math.nan)))

                    c = by_src.setdefault(src, {"DFS":0, "SFD":0, "Control":0, "Total":0})
                    c[reg] += 1; c["Total"] += 1

                if n_dfs < MIN_DFS or n_sfd < MIN_SFD or n_ctrl < MIN_CTRL:
                    continue

                per_metric = {}
                contribs = []; used = 0
                for m in metrics:
                    ad = mae(dfs[m]);  # MAE in DFS
                    as_ = mae(sfd[m])  # MAE in SFD
                    ac = mae(ctrl[m])  # MAE in Control
                    gd = ad - ac if not (math.isnan(ad) or math.isnan(ac)) else float("nan")
                    gs = as_ - ac if not (math.isnan(as_) or math.isnan(ac)) else float("nan")
                    # average-of-available gaps (no min/max)
                    vals_gap = [g for g in (gd, gs) if not math.isnan(g)]
                    if vals_gap:
                        c = sum(vals_gap) / len(vals_gap)
                        contribs.append(c); used += 1
                    else:
                        c = float("nan")
                    per_metric[m] = (ad, as_, ac, gd, gs, c)

                if not contribs:
                    continue

                obj = sum(contribs) / len(contribs)  # objective: avg MAE gap vs control
                results.append((tx_lo, tx_hi, ty_lo, ty_hi, obj, used, n_dfs, n_sfd, n_ctrl, per_metric, by_src))
                print("one donee")

if not results:
    raise SystemExit("No candidate met sample guards; relax MIN_* or widen (lo/hi) ranges.")

# sort ONLY by objective (no tie-breakers)
results.sort(key=lambda z: z[4], reverse=True)
tx_lo, tx_hi, ty_lo, ty_hi, obj, used, ndfs, nsfd, nctrl, per_metric, by_src_best = results[0]

print("\n== Best 4-line thresholds (corners=DFS/SFD, else=Control) ==")
print(f"τx_lo = {tx_lo:.2f}  τx_hi = {tx_hi:.2f}   τy_lo = {ty_lo:.2f}  τy_hi = {ty_hi:.2f}")
print(f"Objective (avg MAE gap) = {obj:.6f}   [metrics used: {used}/{len(metrics)}]")
print(f"Counts: DFS={ndfs}  SFD={nsfd}  Control={nctrl}  Total={len(rows)}")

print("\nPer-metric at best thresholds:")
print("  metric           MAE_DFS       MAE_SFD       MAE_CTRL      gap_DFS      gap_SFD      contrib(avg)")
for m in sorted(per_metric.keys()):
    ad, as_, ac, gd, gs, c = per_metric[m]
    def fmt(v): return f"{v:>10.6f}" if not math.isnan(v) else f"{'nan':>10}"
    print(f"  {m:<15} {fmt(ad)} {fmt(as_)} {fmt(ac)} {fmt(gd)} {fmt(gs)} {fmt(c)}")

print("\nPer-source counts at best thresholds:")
for src in sorted(by_src_best.keys()):
    c = by_src_best[src]
    print(f"  {src:>10}: DFS={c['DFS']:5d}  SFD={c['SFD']:5d}  Control={c['Control']:5d}  Total={c['Total']:5d}")

print("\nTop 5 options by objective (with per-source counts):")
for tx_lo2, tx_hi2, ty_lo2, ty_hi2, obj2, used2, nd2, ns2, nc2, _pm, src_counts in results[:5]:
    print(f"  τx_lo={tx_lo2:.2f} τx_hi={tx_hi2:.2f} | τy_lo={ty_lo2:.2f} τy_hi={ty_hi2:.2f} "
          f"| obj={obj2:.6f} used={used2}/{len(metrics)} | dfs={nd2} sfd={ns2} ctrl={nc2}")
    for src in sorted(src_counts.keys()):
        c = src_counts[src]
        print(f"      - {src:>10}: DFS={c['DFS']:5d}  SFD={c['SFD']:5d}  Control={c['Control']:5d}  Total={c['Total']:5d}")
