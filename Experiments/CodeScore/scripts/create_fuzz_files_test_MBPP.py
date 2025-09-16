#!/usr/bin/env python3
import json
import re
import textwrap
import random
from pathlib import Path
from typing import Optional  # for older Python

INPUT_JSONL = Path("../data/MBPP_test_with_id.jsonl")

# Output relative to scripts/ → Benchmark_Curation/Diff_Fuzzing/CodeScore_Test_1000/MBPP-test
OUT_DIR     = Path("../../../Benchmark_Curation/Diff_Fuzzing/CodeScore_Test_1000") / "MBPP-test"

# Random sampling
SAMPLE_N    = 1000
SEED        = 13

# process only first N rows after sampling (set to None to use all sampled)
PROCESS_N   = None

TEMPLATE = """import atheris
import sys

total = 0
matches = 0
MAX_CASES = 2000

{orig_code_mod}

{gen_code_mod}

def TestOneInput(data):
    global total, matches

    if total >= MAX_CASES:
        raise Exception

    fdp = atheris.FuzzedDataProvider(data)

    try:
        # You must manually constrain input here
        # Example:
        # n = fdp.ConsumeIntInRange(1, 10)
        # arr = [fdp.ConsumeIntInRange(-100, 100) for _ in range(n)]
        # out_a = solve_a(arr, n)
        # out_b = solve_b(arr, n)

        raise NotImplementedError("Insert input logic manually.")

        total += 1
        if out_a == out_b:
            matches += 1
        else:
            print(f"❌ Mismatch: A={{out_a}}, B={{out_b}} | Input: ...")

    except Exception:
        pass

def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput)
    try:
        atheris.Fuzz()
    except Exception:
        pass
    finally:
        print("\\n=== Fuzzing Summary ===")
        print(f"Total test cases: {{total}}")
        print(f"Matches (semantically equivalent): {{matches}}")
        if total > 0:
            print(f"Semantic agreement score: {{matches / total:.2%}}")
        else:
            print("No valid inputs tested.")

if __name__ == "__main__":
    main()
"""

def dedent_strip(code: str) -> str:
    if code is None:
        code = ""
    # normalize tabs → spaces to avoid TabError
    code = code.expandtabs(4)
    return textwrap.dedent(code).strip("\n\r ")

def find_first_def(code: str):
    return re.search(r"(^\s*def\s+)([A-Za-z_]\w*)(\s*\(.*?\)\s*:)", code or "", re.M | re.S)

def rename_first_def_to(code: str, new_name: str) -> Optional[str]:
    m = find_first_def(code)
    if not m:
        return None
    return code[:m.start()] + f"{m.group(1)}{new_name}{m.group(3)}" + code[m.end():]

def extract_param_list(code: str) -> Optional[str]:
    m = re.search(r"^\s*def\s+[A-Za-z_]\w*\s*\((.*?)\)\s*:", code or "", re.M | re.S)
    return m.group(1) if m else None

def wrap_body_as(name: str, params: Optional[str], body: str) -> str:
    body = dedent_strip(body)
    if not body:
        return f"def {name}(*args, **kwargs):\n    pass\n"
    header = f"def {name}({params}):" if params is not None else f"def {name}(*args, **kwargs):"
    return header + "\n" + textwrap.indent(body, "    ") + ("\n" if not body.endswith("\n") else "")

def make_solve_block(src_code: str, target_name: str, fallback_params: Optional[str]):
    code = dedent_strip(src_code)
    renamed = rename_first_def_to(code, target_name)
    if renamed is not None:
        return renamed
    return wrap_body_as(target_name, fallback_params, code)

def main():
    if not INPUT_JSONL.exists():
        raise SystemExit(f"Input not found: {INPUT_JSONL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # random sample
    total_rows = len(rows)
    random.seed(SEED)
    if total_rows > SAMPLE_N:
        rows = random.sample(rows, SAMPLE_N)
        print(f"🎲 Sampled {len(rows)} of {total_rows} rows (seed={SEED}).")
    else:
        print(f"Using all {total_rows} rows (<= {SAMPLE_N}).")

    # optional cap after sampling
    if PROCESS_N is not None:
        rows = rows[:PROCESS_N]
        print(f"⏱  Processing only first {PROCESS_N} sampled rows.")

    created = 0
    for row in rows:
        eid = str(row["id"])
        golden = row.get("golden_code") or ""
        generated = row.get("generated_code") or ""

        gold_params = extract_param_list(golden)

        orig_code_mod = make_solve_block(golden, "solve_a", None)
        gen_code_mod  = make_solve_block(generated, "solve_b", gold_params)

        fuzz_code = TEMPLATE.format(orig_code_mod=orig_code_mod, gen_code_mod=gen_code_mod)

        task_dir = OUT_DIR / eid
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"diff_fuzz_{eid}.py").write_text(fuzz_code, encoding="utf-8")
        created += 1
        print(f"✅ Wrote {task_dir / f'diff_fuzz_{eid}.py'}")

    print(f"\nDone. Created {created} tasks under {OUT_DIR}")
    if PROCESS_N is not None:
        print("Tip: set PROCESS_N = None to process the entire sampled set.")

if __name__ == "__main__":
    main()
