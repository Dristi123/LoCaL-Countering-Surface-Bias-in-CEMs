#!/usr/bin/env python3
import json
import math
from pathlib import Path

# ====== Config ======
INPUT_JSONL = Path("local_with_scores.jsonl")  # <-- change path if needed
OUT_DIR     = Path("worst100_by_metric")
METRICS     = ["codebleu", "crystalbleu", "codebertscore", "codescore"]
TOP_K       = 100
# ====================

def safe_num(x):
    """Return float(x) if it’s a finite number, else None."""
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None

def main():
    if not INPUT_JSONL.exists():
        raise SystemExit(f"Input not found: {INPUT_JSONL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # buckets to collect (metric -> list of (abs_err, row))
    worst = {m: [] for m in METRICS}

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue

            gt = safe_num(row.get("score"))
            if gt is None:
                # can't compute error without ground-truth
                continue

            for m in METRICS:
                mv = safe_num(row.get(m))
                if mv is None:
                    continue
                abs_err = abs(mv - gt)
                worst[m].append((abs_err, row))

    # sort, take top-K, and write files
    for m in METRICS:
        # sort by error descending (largest error = worst)
        sorted_rows = sorted(worst[m], key=lambda t: t[0], reverse=True)
        top = sorted_rows[:TOP_K] if len(sorted_rows) > TOP_K else sorted_rows

        out_path = OUT_DIR / f"worst_top{TOP_K}_{m}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_f:
            for rank, (abs_err, row) in enumerate(top, start=1):
                # write original row plus a few helper fields
                enriched = dict(row)
                enriched[f"{m}_abs_err_vs_score"] = abs_err
                enriched["_worst_metric"] = m
                enriched["_worst_rank"] = rank
                out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

        print(f"Wrote {len(top):3d} rows to {out_path}")

if __name__ == "__main__":
    main()
