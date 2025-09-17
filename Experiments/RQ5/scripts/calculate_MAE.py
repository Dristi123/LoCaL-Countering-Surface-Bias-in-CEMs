
import json, math, csv
from pathlib import Path

RES_PARENT = Path(".")
PREFIX = "cs_infer_results_"
PRED_KEYS = ("prediction", "pred_score", "predict_score")  

def latest_results_dir(parent: Path) -> Path:
    cands = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith(PREFIX)])
    if not cands:
        raise SystemExit(f"No '{PREFIX}*' under {parent.resolve()}")
    return cands[-1]

def fnum(x):
    try: return float(x)
    except: return None

def file_abs_errs(p: Path):
    s, n = 0.0, 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue
            y = fnum(obj.get("score"))
            if y is None: 
                continue
            pred = None
            for k in PRED_KEYS:
                if k in obj:
                    pred = fnum(obj.get(k))
                    if pred is not None:
                        break
            if pred is None:
                continue
            s += abs(y - pred); n += 1
    return s, n

def main():
    root = latest_results_dir(RES_PARENT)
    out_csv = RES_PARENT / "mae_local_vs_baseline.csv"

    rows = {}
    for p in sorted(root.rglob("*.jsonl")):
        parts = p.relative_to(root).parts
        if len(parts) < 3:
            continue
        exp, label = parts[0], parts[1].lower()
        if label not in ("local", "baseline"):
            continue

        ck = Path(parts[-1]).stem.replace("run_", "run") 
        exp_id = f"{ck}_{exp}"

        s, n = file_abs_errs(p)
        d = rows.setdefault(exp_id, {"local_s":0.0, "local_n":0, "base_s":0.0, "base_n":0})
        if label == "local":
            d["local_s"] += s; d["local_n"] += n
        else:
            d["base_s"]  += s; d["base_n"]  += n

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["exp", "mae_local", "mae_baseline"])
        for exp_id in sorted(rows):
            d = rows[exp_id]
            mae_local = (d["local_s"]/d["local_n"]) if d["local_n"] else math.nan
            mae_base  = (d["base_s"]/d["base_n"])   if d["base_n"]   else math.nan
            w.writerow([exp_id,
                        f"{mae_local:.6f}" if math.isfinite(mae_local) else "NaN",
                        f"{mae_base:.6f}"  if math.isfinite(mae_base)  else "NaN"])

    print(f"Output stored in {out_csv.resolve()}")

if __name__ == "__main__":
    main()
