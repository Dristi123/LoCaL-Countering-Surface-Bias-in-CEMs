#!/usr/bin/env python3
import csv, math, re
from collections import defaultdict
from pathlib import Path

IN  = Path("mae_results.csv")         
OUT = Path("../results/mae_table_by_percent.csv")
TOTAL = 2026                                
BUCKET = 10                                  

def pct_from_exp(s):
    m = re.search(r"_([0-9]+)$", str(s).strip())
    if not m: return None
    v = int(m.group(1))
    p = v if 0 <= v <= 100 else v * 100.0 / TOTAL
    return max(0, min(100, int(round(p/BUCKET)*BUCKET)))

def fnum(x):
    try:
        v = float(x);  return v if math.isfinite(v) else None
    except: return None

def mean_std(vals):
    n = len(vals)
    if n == 0: return None, None
    m = sum(vals)/n
    if n == 1: return m, 0.0
    return m, (sum((x-m)**2 for x in vals)/(n-1))**0.5

def fm(x): return "" if x is None else f"{x:.2f}"
def fs(x): return "" if x is None else f"±{x:.2f}"

local = defaultdict(list); base = defaultdict(list); combo = defaultdict(list)
with IN.open("r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        p = pct_from_exp(row["exp"]);  
        if p is None: continue
        ml = fnum(row["mae_local"]); mb = fnum(row["mae_baseline"])
        if ml is not None: local[p].append(ml)
        if mb is not None: base[p].append(mb)
        if ml is not None and mb is not None: combo[p].append((ml+mb)/2)

cols = list(range(0, 101, 10))
rows = [["Dataset","p (%)"] + [str(c) for c in cols]]
for name, bucket in (("Combined", combo), ("Baseline", base), ("LoCaLtest", local)):
    means = [fm(mean_std(bucket[c])[0]) for c in cols]
    stds  = [f"({fs(mean_std(bucket[c])[1])})" for c in cols]
    rows.append([name,""] + means)
    rows.append(["(Std.)",""] + stds)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerows(rows)
