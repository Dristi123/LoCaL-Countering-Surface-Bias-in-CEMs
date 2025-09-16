#!/usr/bin/env python3
import sys, os, json, math, shutil, subprocess, time, random
from pathlib import Path

PIE_JSONL = Path("../../../Source_Benchmarks/PIE/test.jsonl")
LOCAL_JSONL = Path("../../../LoCaL.jsonl")
INFERENCE_PY = Path("inference.py")

K = 100
N_RUNS = 10
SEED_BASE = 1337
SKIP_CODESCORE = False

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs_temp"

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run_cmd(cmd, cwd=None, env=None, timeout=None):
    res = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if res.returncode != 0:
        print(res.stdout)
        raise SystemExit(f"{' '.join(cmd)} failed")
    return res.stdout

def _read_pie_candidates(rows):
    out = []
    for r in rows:
        c1 = r.get("input"); c2 = r.get("target"); pid = r.get("problem_id")
        if c1 and c2 and pid:
            out.append({"problem_id": pid, "golden_code": c1, "generated_code": c2})
    return out

def _pick_pos_from_pie(pie_rows, k, rng):
    cands = _read_pie_candidates(pie_rows)
    if len(cands) < k:
        raise SystemExit(f"not enough PIE pairs: {len(cands)} < {k}")
    return rng.sample(cands, k)

def _build_mismatch_negs(pos_rows, rng):
    negs = []
    n = len(pos_rows)
    for i in range(n):
        pid_i = pos_rows[i]["problem_id"]
        g = pos_rows[i]["golden_code"]
        candidates = [j for j in range(n) if j != i and pos_rows[j]["problem_id"] != pid_i]
        if not candidates:
            candidates = [j for j in range(n) if j != i]
        j = rng.choice(candidates)
        h = pos_rows[j]["generated_code"]
        negs.append({"golden_code": g, "generated_code": h, "score": 0.0})
    return negs

def _sample_mutant_negs(local_rows, k, rng):
    cands = []
    for r in local_rows:
        vt = (r.get("variant_type") or "").lower()
        try:
            sc = float(r.get("df_score", 1))
        except Exception:
            continue
        g = r.get("original_code"); h = r.get("variant_code")
        if vt == "mutation" and sc == 0.0 and g and h:
            cands.append({"golden_code": g, "generated_code": h, "score": 0.0})
    if len(cands) < k:
        raise SystemExit(f"not enough LOCAL MUT negatives: {len(cands)} < {k}")
    return rng.sample(cands, k)

def build_datasets(run_dir: Path, seed: int):
    rng = random.Random(seed)
    pie_rows = load_jsonl(PIE_JSONL)
    local_rows = load_jsonl(LOCAL_JSONL)
    if not pie_rows:
        raise SystemExit(f"no rows in {PIE_JSONL}")
    if not local_rows:
        raise SystemExit(f"no rows in {LOCAL_JSONL}")
    pos = _pick_pos_from_pie(pie_rows, K, rng)
    pos_rows = [{"golden_code": r["golden_code"], "generated_code": r["generated_code"], "score": 1.0} for r in pos]
    neg_mismatch = _build_mismatch_negs(pos, rng)
    write_jsonl(run_dir / "ds_orig.jsonl", pos_rows + neg_mismatch)
    neg_mutants = _sample_mutant_negs(local_rows, K, rng)
    write_jsonl(run_dir / "ds_replaced.jsonl", pos_rows + neg_mutants)

def one_run(run_idx: int):
    run_dir = RUNS_DIR / f"run_{run_idx:03d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    seed = SEED_BASE + run_idx
    build_datasets(run_dir, seed)

    ds_orig_in  = run_dir / "ds_orig.jsonl"
    ds_repl_in  = run_dir / "ds_replaced.jsonl"
    ds_orig_out = run_dir / "ds_orig_with_scores.jsonl"
    ds_repl_out = run_dir / "ds_replaced_with_scores.jsonl"

    infer_path = SCRIPT_DIR / INFERENCE_PY
    if not infer_path.exists():
        raise SystemExit(f"missing {infer_path}")

    base = [sys.executable, str(infer_path)]
    if SKIP_CODESCORE:
        base += ["--skip-codescore"]

    run_cmd(base + ["--in", str(ds_orig_in), "--out", str(ds_orig_out)], cwd=SCRIPT_DIR)
    run_cmd(base + ["--in", str(ds_repl_in), "--out", str(ds_repl_out)], cwd=SCRIPT_DIR)

    print(f"run {run_idx:03d} done: {run_dir}")

def main():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(N_RUNS):
        one_run(i)
    print("done")

if __name__ == "__main__":
    main()
