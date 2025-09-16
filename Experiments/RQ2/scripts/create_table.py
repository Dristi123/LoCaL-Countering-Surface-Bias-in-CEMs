
import json, math
from pathlib import Path
from statistics import fmean, stdev
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

from pathlib import Path

RUNS_DIR = Path("runs")
RESULTS_DIR = Path("../results")
OUT_CSV = RESULTS_DIR / "distinguishability.csv"

METRICS = [
    ("codebleu",      "CodeBLEU"),
    ("crystalbleu",   "CrystalBLEU"),
    ("codebertscore", "CodeBERTScore"),
    ("codescore",     "CodeScore"),
]

def class_means(rows, key):
    eq, ne = [], []
    for r in rows:
        y = r.get("score")
        try:
            v = float(r.get(key))
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        if y in (1, 1.0, True):
            eq.append(v)
        elif y in (0, 0.0, False):
            ne.append(v)
    mu_eq = fmean(eq) if eq else float("nan")
    mu_ne = fmean(ne) if ne else float("nan")
    return mu_eq, mu_ne

def mean_std(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return fmean(xs), stdev(xs)

def r2(x):
    if not math.isfinite(x):
        return "NA"
    try:
        return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return "NA"

def fmt(mu, sd):
    if not math.isfinite(mu):
        return "NA"
    if not math.isfinite(sd):
        sd = 0.0
    return f"{r2(mu)}(±{r2(sd)})"

run_dirs = sorted([p for p in RUNS_DIR.glob("run_*") if p.is_dir()])
if not run_dirs:
    raise SystemExit(f"[error] no runs found in {RUNS_DIR}")

per_metric = {
    pretty: {
        "eq_runs": [],
        "orig_neq_runs": [], "orig_d_runs": [],
        "repl_neq_runs": [], "repl_d_runs": [],
    }
    for _, pretty in METRICS
}

for rd in run_dirs:
    p_orig = rd / "ds_orig_with_scores.jsonl"
    p_repl = rd / "ds_replaced_with_scores.jsonl"
    if not (p_orig.exists() and p_repl.exists()):
        continue
    rows_o = [json.loads(s) for s in p_orig.read_text(encoding="utf-8").splitlines() if s.strip()]
    rows_r = [json.loads(s) for s in p_repl.read_text(encoding="utf-8").splitlines() if s.strip()]

    for key, pretty in METRICS:
        mu_eq_o, mu_neq_o = class_means(rows_o, key)
        mu_eq_r, mu_neq_r = class_means(rows_r, key)

        eq_parts = [x for x in (mu_eq_o, mu_eq_r) if math.isfinite(x)]
        if eq_parts:
            per_metric[pretty]["eq_runs"].append(fmean(eq_parts))
        if math.isfinite(mu_neq_o):
            per_metric[pretty]["orig_neq_runs"].append(mu_neq_o)
        if math.isfinite(mu_neq_r):
            per_metric[pretty]["repl_neq_runs"].append(mu_neq_r)
        if math.isfinite(mu_eq_o) and math.isfinite(mu_neq_o) and mu_neq_o != 0:
            per_metric[pretty]["orig_d_runs"].append(mu_eq_o / mu_neq_o)
        if math.isfinite(mu_eq_r) and math.isfinite(mu_neq_r) and mu_neq_r != 0:
            per_metric[pretty]["repl_d_runs"].append(mu_eq_r / mu_neq_r)

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    f.write("CEM,EQ Score,ds_orig NEQ Score,ds_orig d,ds_replaced NEQ Score,ds_replaced d\n")
    for _, pretty in METRICS:
        eq_mu, eq_sd = mean_std(per_metric[pretty]["eq_runs"])
        o_neq_mu, o_neq_sd = mean_std(per_metric[pretty]["orig_neq_runs"])
        o_d_mu, o_d_sd = mean_std(per_metric[pretty]["orig_d_runs"])
        r_neq_mu, r_neq_sd = mean_std(per_metric[pretty]["repl_neq_runs"])
        r_d_mu, r_d_sd = mean_std(per_metric[pretty]["repl_d_runs"])
        f.write(",".join([
            pretty,
            fmt(eq_mu, eq_sd),
            fmt(o_neq_mu, o_neq_sd),
            fmt(o_d_mu, o_d_sd),
            fmt(r_neq_mu, r_neq_sd),
            fmt(r_d_mu, r_d_sd),
        ]) + "\n")

print(f"Results stored in {OUT_CSV}")
