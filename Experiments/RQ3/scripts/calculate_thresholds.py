#!/usr/bin/env python3
import json, math
from pathlib import Path
from statistics import median
INPUT = Path("combined_custom_sample.jsonl")
SEM, SURF = "score", "surfaceSim"
METRICS = ["codebleu", "codebertscore", "crystalbleu", "codescore"]

STEP = 0.05


def f2(x):
    try: return float(x)
    except: return float("nan")


def mae(pairs):
    xs = [abs(s - m) for s, m in pairs if not (math.isnan(s) or math.isnan(m))]
    return (sum(xs)/len(xs)) if xs else math.nan


def vals(lo, hi, step):
    n = int(round((hi - lo)/step))
    return [round(lo + i*step, 2) for i in range(n+1)]

def infer_source(obj):
    rid = str(obj.get("id", ""))
    if rid.startswith("LOCAL"): return "local"
    return "baseline"
lo_r=0.10
hi_r=0.10
mid=0.5
TX_LOS = vals(0.10, 0.90, STEP)
TX_HIS = vals(0.10, 0.90, STEP)
TY_LOS = vals(0.10, 0.90, STEP)
TY_HIS = vals(0.10, 0.90, STEP)

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


MIN_DFS, MIN_SFD, MIN_CTRL = 50, 50, 150
results = []  
for tx_lo in TX_LOS:
    for tx_hi in TX_HIS:
        for ty_lo in TY_LOS:
            for ty_hi in TY_HIS:
                if tx_hi <= tx_lo or ty_hi <= ty_lo or tx_lo>(mid+lo_r+STEP) or ty_hi>=(1.0-hi_r+STEP) or ty_lo>(mid+lo_r+STEP) or tx_hi>=(1.0-hi_r+STEP):
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
                    ad = mae(dfs[m]);  
                    as_ = mae(sfd[m])  
                    ac = mae(ctrl[m])  
                    gd = ad - ac if not (math.isnan(ad) or math.isnan(ac)) else float("nan")
                    gs = as_ - ac if not (math.isnan(as_) or math.isnan(ac)) else float("nan")
                   
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
                #print("one donee")

if not results:
    raise SystemExit("No candidate met sample guards; relax MIN_* or widen (lo/hi) ranges.")

def _tie_key(z):
    tx_lo, tx_hi, ty_lo, ty_hi, obj, *_ = z
    o2 = round(obj, 2)
    return (o2, ty_hi, -ty_lo, tx_hi, -tx_lo)

results.sort(key=_tie_key, reverse=True)
tx_lo, tx_hi, ty_lo, ty_hi, obj, used, ndfs, nsfd, nctrl, per_metric, by_src_best = results[0]



print("\n== Best 4-line thresholds (corners=DFS/SFD, else=Control) ==")
print(f"τx_lo = {tx_lo:.2f}  τx_hi = {tx_hi:.2f}   τy_lo = {ty_lo:.2f}  τy_hi = {ty_hi:.2f}")
print(f"Objective (avg MAE gap) = {obj:.6f}   [metrics used: {used}/{len(metrics)}]")
print(f"Counts: DFS={ndfs}  SFD={nsfd}  Control={nctrl}  Total={len(rows)}")



print("\nPer-source counts at best thresholds:")
for src in sorted(by_src_best.keys()):
    c = by_src_best[src]
    print(f"  {src:>10}: DFS={c['DFS']:5d}  SFD={c['SFD']:5d}  Control={c['Control']:5d}  Total={c['Total']:5d}")
