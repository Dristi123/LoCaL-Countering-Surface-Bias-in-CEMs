
import json, math
from pathlib import Path

BASELINE_CS = Path("baseline_codescore_with_scores.jsonl")
LOCAL_ALL   = Path("local_with_scores.jsonl")
SHARECODE   = Path("baseline_sharecode_with_scores.jsonl")

OUT_CSV = Path("../results/cem_mae_table.csv")

METRICS = ["codebleu","crystalbleu","codebertscore","codescore"]

def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def mae_for_file(p: Path, keep=None):
    mae_sum = {m:0.0 for m in METRICS}
    cnt     = {m:0   for m in METRICS}
    if not p.exists():
        return {m: None for m in METRICS}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                r = json.loads(s)
            except Exception:
                continue
            if keep is not None and not keep(r):
                continue
            gt = fnum(r.get("score"))
            if gt is None:
                continue
            for m in METRICS:
                mv = fnum(r.get(m))
                if mv is None:
                    continue
                d = mv - gt
                mae_sum[m] += abs(d)
                cnt[m]     += 1
    out = {}
    for m in METRICS:
        out[m] = (mae_sum[m]/cnt[m]) if cnt[m] > 0 else None
    return out

def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) and math.isfinite(x) else "N/A"

def main():
    py_codescore = mae_for_file(BASELINE_CS)
    local_all    = mae_for_file(LOCAL_ALL)
    local_opt    = mae_for_file(LOCAL_ALL, keep=lambda r: "OP" in str(r.get("id","")).upper())
    local_mut    = mae_for_file(LOCAL_ALL, keep=lambda r: "MUT" in str(r.get("id","")).upper())
    java_share   = mae_for_file(SHARECODE)

    rows = [
        ("Python","CodeScore_test", py_codescore),
        ("Python","LoCaL",         local_all),
        ("Python","LoCaLopt",      local_opt),
        ("Python","LoCaLmut",      local_mut),
        ("Java","ShareCode_java",  java_share),
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8") as f:
        f.write("Language,Dataset,CodeBLEU,CrystalBLEU,CodeBERTScore,CodeScore\n")
        for lang, ds, scores in rows:
            f.write(",".join([
                lang, ds,
                fmt(scores.get("codebleu")),
                fmt(scores.get("crystalbleu")),
                fmt(scores.get("codebertscore")),
                fmt(scores.get("codescore")),
            ]) + "\n")

if __name__ == "__main__":
    main()
