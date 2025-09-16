#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN_JSONL = Path("PIE_RQ1_with_scores.jsonl")
OUT_CSV = Path("../results/spearman_surfaceSim_vs_metrics.csv")
METRICS = ["codebleu", "crystalbleu", "codebertscore", "codescore"]

df = pd.read_json(IN_JSONL, lines=True)
cols = ["surfaceSim"] + METRICS
for col in cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "surfaceSim" not in df.columns:
    raise RuntimeError("surfaceSim column not found.")

EPS_POS = np.nextafter(0, 1)
def clip_p(p):
    if not np.isfinite(p) or p <= 0.0:
        return EPS_POS
    return p

rows = []
for key in METRICS:
    if key not in df.columns:
        continue
    sub = df[["surfaceSim", key]].dropna()
    if sub.empty:
        continue
    x = sub["surfaceSim"].to_numpy(float)
    y = sub[key].to_numpy(float)
    rho, p = spearmanr(x, y)
    rows.append({"metric": key, "n": len(sub), "spearman_rho": float(rho), "p_value": float(clip_p(p))})

out = pd.DataFrame(rows, columns=["metric", "n", "spearman_rho", "p_value"])
out.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(out)} rows to {OUT_CSV}")
