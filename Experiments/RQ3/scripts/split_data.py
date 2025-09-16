#!/usr/bin/env python3
import json, random
from pathlib import Path


BASELINE = Path("baseline_codescore_with_scores.jsonl")
LOCAL = Path("local_with_scores.jsonl")
OUT      = Path("combined_dataset.jsonl")
SEED     = 1337


N_BASELINE = 854           
N_LOCAL_OP = 427       
N_LOCAL_MUT =  427    


def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                rows.append(json.loads(s))
            except Exception:
                pass
    return rows

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def is_op(row):
   
    rid = str(row.get("id", "")).upper()
    if "_OP_" in rid or rid.endswith("_OP"):
        return True
    sc = row.get("score")
    try:
        return float(sc) >= 0.5
    except Exception:
        return False

  

def main():
    if not BASELINE.exists() or not LOCAL.exists():
        raise SystemExit("Missing input files.")

    baseline = load_jsonl(BASELINE)
    local    = load_jsonl(LOCAL)
    if not baseline or not local:
        raise SystemExit(f"Empty inputs: baseline={len(baseline)}, local={len(local)}")

 
    local_ops  = [r for r in local if is_op(r)]
    local_muts = [r for r in local if not is_op(r)]

  
    n_b  = min(N_BASELINE, len(baseline))
    n_op = N_LOCAL_OP if N_LOCAL_OP is not None else n_b // 2
    n_mt = N_LOCAL_MUT if N_LOCAL_MUT is not None else n_b // 2

    

    random.seed(SEED)
    samp_baseline = random.sample(baseline, n_b)
    samp_ops      = random.sample(local_ops, n_op) if n_op > 0 else []
    samp_muts     = random.sample(local_muts, n_mt) if n_mt > 0 else []

   
    for r in samp_baseline:
        r.setdefault("source", "baseline")
        r.setdefault("local_type", "baseline")
    for r in samp_ops:
        r.setdefault("source", "local")
        r.setdefault("local_type", "OP")
    for r in samp_muts:
        r.setdefault("source", "local")
        r.setdefault("local_type", "MUT")

    merged = samp_baseline + samp_ops + samp_muts
    random.shuffle(merged)
    write_jsonl(OUT, merged)

    print(f"wrote {len(merged)} -> {OUT.resolve()}")
    print(f"   sampled baseline : {len(samp_baseline)}")
    print(f"   sampled local OP : {len(samp_ops)}")
    print(f"   sampled local MUT: {len(samp_muts)}")

if __name__ == "__main__":
    main()
