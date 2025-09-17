
import csv, math, re
from collections import defaultdict
from pathlib import Path

IN  = Path("mae_results.csv")                
OUT = Path("../results/mae_table_by_percent.csv")

PCT_RE = re.compile(r"_([0-9]{1,3})$")       
COLS   = list(range(0, 101, 10))            

def parse_pct(exp: str):
    m = PCT_RE.search(exp.strip())
    if not m:
        return None
    p = int(m.group(1))
    return p if 0 <= p <= 100 else None

def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except:
        return None

def mean_std(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None, None
    m = sum(vals)/n
    if n == 1:
        return m, 0.0
    var = sum((x - m)**2 for x in vals)/(n - 1)
    return m, var**0.5

def fm(x): return "" if x is None else f"{x:.2f}"
def fs(x): return "" if x is None else f"±{x:.2f}"

local  = defaultdict(list)
base   = defaultdict(list)
combo  = defaultdict(list)

with IN.open("r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        p = parse_pct(row["exp"])
        if p is None:
            continue
        ml = fnum(row.get("mae_local"))
        mb = fnum(row.get("mae_baseline"))
        if ml is not None:
            local[p].append(ml)
        if mb is not None:
            base[p].append(mb)
        if ml is not None and mb is not None:
            combo[p].append((ml + mb) / 2.0)

rows = [["Dataset", "p (%)"] + [str(c) for c in COLS]]
for name, bucket in (("Combined", combo), ("Baseline", base), ("LoCaLtest", local)):
    means = []
    stds  = []
    for c in COLS:
        m, s = mean_std(bucket.get(c, []))
        means.append(fm(m))
        stds.append(f"({fs(s)})")
    rows.append([name, ""] + means)
    rows.append(["(Std.)", ""] + stds)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as f:
    csv.writer(f).writerows(rows)

print(f"Outputs stored in {OUT.resolve()}")
