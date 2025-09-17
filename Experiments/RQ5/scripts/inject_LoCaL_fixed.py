#!/usr/bin/env python3
import json, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ----------------- CONFIG -----------------
SEED = 2025
random.seed(SEED)

N_LOCAL        = 2000           # how many LoCaL rows to use for training
RANDOM_SAMPLE  = True           # True: random sample; False: 50/50 OP/MUT balanced (tops up if one side short)
VAL_RATIO      = 0.125          # validation size = 12.5% of corresponding train
INCLUDE_PIE    = False          # exclude PIE items from LoCaL if False
# ------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

LOCAL_PATH = SCRIPT_DIR.parent.parent / "Common_Scripts" / "LoCaL_with_PIEextra_mutOnly.jsonl"

BASE_CS_DIR = Path("../../CodeScore/CodeScore/data")
BASE_TRAIN  = BASE_CS_DIR / "train_base_1000_2000_merged.jsonl"
BASE_VAL    = BASE_CS_DIR / "val_base_1000_2000_merged.jsonl"

APPS_ORIG = Path("../../Datasets/APPS/apps_orig.jsonl")
HE_ORIG   = Path("../Datasets/HumanEval/HE_orig.jsonl")
MBPP_ORIG = Path("../Datasets/MBPP/MBPP_sorted_10pct.jsonl")

# Include "Random_" in outdir name when RANDOM_SAMPLE is True
OUT_DIR = SCRIPT_DIR / f"merged_splits_local_run3_{'Random_' if RANDOM_SAMPLE else ''}{N_LOCAL if N_LOCAL is not None else 'ALL'}"

# -------------- IO utils --------------
def read_jsonl(p: Path) -> List[Dict]:
    rows = []
    if not p.exists(): return rows
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                pass
    return rows

def write_jsonl(p: Path, rows: List[Dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def pair_key(r: Dict) -> Tuple[str, str]:
    return (r.get("golden_code",""), r.get("generated_code",""))


def load_orig_map(path: Path, dataset: str) -> Dict[str, str]:
    m: Dict[str, str] = {}
    if not path.exists(): return m
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = obj.get("task_id"); txt = obj.get("text")
            if tid is None or not isinstance(txt, str): continue
            tid_str = str(tid).strip()
            if not tid_str: continue
            m[tid_str] = txt
            if dataset == "HE":
                if tid_str.startswith("HumanEval_"):
                    suf = tid_str.split("HumanEval_", 1)[1]
                    if suf and suf not in m: m[suf] = txt
                elif tid_str.isdigit():
                    he = f"HumanEval_{tid_str}"
                    if he not in m: m[he] = txt
    return m

def get_desc_maps() -> Dict[str, Dict[str, str]]:
    return {
        "APPS": load_orig_map(APPS_ORIG, "APPS"),
        "HE":   load_orig_map(HE_ORIG,   "HE"),
        "MBPP": load_orig_map(MBPP_ORIG, "MBPP"),
    }

def lookup_desc(origin: str, task_id: str, maps: Dict[str, Dict[str, str]]) -> str:
    t = str(task_id).strip()
    m = maps.get(origin.strip(), {})
    if t in m: return m[t]
    if origin == "HE":
        if t.isdigit() and f"HumanEval_{t}" in m: return m[f"HumanEval_{t}"]
        if t.startswith("HumanEval_"):
            suf = t.split("HumanEval_", 1)[1]
            return m.get(suf, "")
    return ""


def convert_local_to_base(e: Dict, maps: Dict[str, Dict[str, str]]) -> Dict:
    origin  = str(e.get("origin","")).strip()
    task_id = str(e.get("task_id","")).strip()
    variant = str(e.get("variant_id","")).strip()
    gid = f"{origin}_{task_id}_{variant}" if variant else f"{origin}_{task_id}"
    try:
        score = float(e.get("df_score", 0.0))  # df_score provided in your file
    except Exception:
        score = 0.0
    return {
        "id": gid,
        "golden_code":    e.get("original_code",""),
        "generated_code": e.get("variant_code",""),
        "score": score,
        "source": lookup_desc(origin, task_id, maps),
    }

def is_op(row: Dict) -> bool:
    i = row.get("id","").upper()
    return ("_OP_" in i) or i.endswith("_OP")

def is_mut(row: Dict) -> bool:
    i = row.get("id","").upper()
    return ("_MUT_" in i) or i.endswith("_MUT")

def pick_balanced(ops_list: List[Dict], muts_list: List[Dict], k: int) -> List[Dict]:
    if k <= 0: return []
    want_ops  = (k + 1) // 2
    want_muts = k // 2
    take_ops  = min(want_ops,  len(ops_list))
    take_muts = min(want_muts, len(muts_list))
    rem = k - (take_ops + take_muts)
    if rem > 0:
        ops_left  = len(ops_list)  - take_ops
        muts_left = len(muts_list) - take_muts
        if ops_left >= muts_left:
            add = min(rem, ops_left);  take_ops += add;  rem -= add
        if rem > 0:
            add = min(rem, muts_left); take_muts += add; rem -= add
    chosen = []
    if take_ops  > 0: chosen += random.sample(ops_list,  take_ops)
    if take_muts > 0: chosen += random.sample(muts_list, take_muts)
    random.shuffle(chosen)
    return chosen

def sample_local(pool: List[Dict], k: int, balanced: bool) -> List[Dict]:
    if k <= 0 or not pool: return []
    k = min(k, len(pool))
    if balanced:
        ops  = [r for r in pool if is_op(r)]
        muts = [r for r in pool if is_mut(r)]
        return pick_balanced(ops, muts, k)
    return random.sample(pool, k)

def count_local(rows: List[Dict]) -> int:
    return sum(1 for r in rows if is_op(r) or is_mut(r))


def build_val(train_rows: List[Dict],
              base_val: List[Dict],
              local_pool_for_val: List[Dict],
              target: int,
              balanced_local: bool,
              p_local: Optional[float] = None) -> List[Dict]:
  
    if target <= 0:
        return []

    if p_local is None:
        local_need = min(target // 2, len(local_pool_for_val))
        base_need  = target - local_need
    else:
        p_local = max(0.0, min(1.0, p_local))
        local_need = int(round(target * p_local))
        base_need  = target - local_need

        if local_need > len(local_pool_for_val):
            local_need = len(local_pool_for_val)
            base_need  = min(target - local_need, len(base_val))
        if base_need > len(base_val):
            base_need  = len(base_val)
            local_need = min(target - base_need, len(local_pool_for_val))


    if len(base_val) > 0 and len(local_pool_for_val) > 0 and target >= 2:
        if local_need == 0:
            local_need = 1; base_need = max(0, target - local_need)
        elif base_need == 0:
            base_need = 1; local_need = max(0, target - base_need)

    base_part  = random.sample(base_val, base_need) if base_need > 0 else []

    used_pairs = {pair_key(r) for r in (train_rows + base_part)}
    local_clean = [r for r in local_pool_for_val if pair_key(r) not in used_pairs]
    local_part  = sample_local(local_clean, min(local_need, len(local_clean)), balanced_local)

    out = base_part + local_part


    if len(out) < target:
        need = target - len(out)
        leftovers_local = [r for r in local_clean if r not in local_part]
        extra_local = sample_local(leftovers_local, min(need, len(leftovers_local)), balanced_local)
        out += extra_local
        need = target - len(out)
        if need > 0:
            base_leftovers = [r for r in base_val if r not in base_part]
            out += base_leftovers[:need]

    if len(out) > target:
        random.shuffle(out)
        out = out[:target]
    return out


def main():
 
    base_train = read_jsonl(BASE_TRAIN)
    base_val   = read_jsonl(BASE_VAL)

   
    maps = get_desc_maps()
    local_raw = read_jsonl(LOCAL_PATH)
    local_all: List[Dict] = []
    for e in local_raw:
        if not INCLUDE_PIE and str(e.get("origin","")).strip() == "PIE":
            continue
        oc = e.get("original_code"); vc = e.get("variant_code")
        if not isinstance(oc, str) or not isinstance(vc, str): continue
        local_all.append(convert_local_to_base(e, maps))

    seen_pairs = {pair_key(r) for r in (base_train + base_val)}
    pool: List[Dict] = []
    for r in local_all:
        k = pair_key(r)
        if k in seen_pairs: continue
        seen_pairs.add(k)
        pool.append(r)

    if not pool:
        print("[error] No LoCaL candidates available after dedupe.")
        return


    k_train = min(N_LOCAL, len(pool))
    selected_train_local = sample_local(pool, k_train, balanced=(not RANDOM_SAMPLE))
    train_append = base_train + selected_train_local

    p_local_append = len(selected_train_local) / max(1, len(train_append))

 
    replace_k = min(len(base_train), len(selected_train_local))
    if replace_k > 0:
        drop_idx = set(random.sample(range(len(base_train)), replace_k))
        base_keep = [r for i, r in enumerate(base_train) if i not in drop_idx]
        train_replace = base_keep + selected_train_local[:replace_k]
    else:
        train_replace = base_train[:]


    p_local_replace = replace_k / max(1, len(train_replace))

    train_pairs_append = {pair_key(r) for r in selected_train_local}
    pool_for_val_append = [r for r in pool if pair_key(r) not in train_pairs_append]


    target_val_append = int(round(VAL_RATIO * len(train_append)))
    val_local_append = build_val(
        train_rows=train_append,
        base_val=base_val,
        local_pool_for_val=pool_for_val_append,
        target=target_val_append,
        balanced_local=(not RANDOM_SAMPLE),
        p_local=p_local_append, 
    )


    train_pairs_replace = {pair_key(r) for r in selected_train_local[:replace_k]}
    pool_for_val_replace = [r for r in pool if pair_key(r) not in train_pairs_replace]


    target_val_replace = int(round(VAL_RATIO * len(train_replace)))
    val_local_replace = build_val(
        train_rows=train_replace,
        base_val=base_val,
        local_pool_for_val=pool_for_val_replace,
        target=target_val_replace,
        balanced_local=(not RANDOM_SAMPLE),
        p_local=p_local_replace,  
    )


    random.shuffle(train_append)
    random.shuffle(train_replace)
    random.shuffle(val_local_append)
    random.shuffle(val_local_replace)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train_append.jsonl",      train_append)
    write_jsonl(OUT_DIR / "val_local_append.jsonl",  val_local_append)
    write_jsonl(OUT_DIR / "train_replace.jsonl",     train_replace)
    write_jsonl(OUT_DIR / "val_local_replace.jsonl", val_local_replace)
    write_jsonl(OUT_DIR / "val_base.jsonl",          base_val)  # reference


    ta_loc = count_local(train_append);   ta_tot = len(train_append)
    tr_loc = count_local(train_replace);  tr_tot = len(train_replace)
    va_loc = count_local(val_local_append);  va_tot = len(val_local_append)
    vr_loc = count_local(val_local_replace); vr_tot = len(val_local_replace)

    print(f"[append ] train={ta_tot:5d} (local={ta_loc}, {ta_loc/ta_tot:6.2%}) | "
          f"val(target={target_val_append}) -> {va_tot:4d} (local={va_loc}, { (va_loc/va_tot if va_tot else 0):6.2%})")
    print(f"[replace] train={tr_tot:5d} (local={tr_loc}, {tr_loc/tr_tot:6.2%}) | "
          f"val(target={target_val_replace}) -> {vr_tot:4d} (local={vr_loc}, { (vr_loc/vr_tot if vr_tot else 0):6.2%})")
    print(f"[config ] N_LOCAL={N_LOCAL} | RANDOM_SAMPLE={RANDOM_SAMPLE} | VAL_RATIO={VAL_RATIO} | INCLUDE_PIE={INCLUDE_PIE}")
    print(f"[out    ] {OUT_DIR}")

if __name__ == "__main__":
    main()
