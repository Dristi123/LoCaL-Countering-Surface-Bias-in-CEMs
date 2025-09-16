#!/usr/bin/env python3
from pathlib import Path

# -------- Fuzz script template --------
fuzz_code_template = """import atheris
import sys
import subprocess

total = 0
matches = 0
MAX_CASES = 1000

script_a = "gold.py"
script_b = "gen.py"

def run_program(script_path, input_str):
    try:
        result = subprocess.run(
            ["python3", script_path],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=45
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception:
        return "ERROR"

def TestOneInput(data):
    global total, matches
    if total >= MAX_CASES:
        raise Exception("Reached max test cases")

    fdp = atheris.FuzzedDataProvider(data)

    try:
        # You must manually constrain input here
        # Example for a single integer input:
        # val = fdp.ConsumeIntInRange(1, 100000)
        # input_str = f"{val}\\n"

        raise NotImplementedError("Insert input logic manually.")

        out_a = run_program(script_a, input_str)
        out_b = run_program(script_b, input_str)

        total += 1
        if out_a == out_b:
            matches += 1
        else:
            print(f"❌ Mismatch: A={out_a}, B={out_b} | Input: ...")

    except Exception:
        pass

def main():
    atheris.Setup(sys.argv, TestOneInput)
    try:
        atheris.Fuzz()
    except Exception:
        pass
    finally:
        print("\\n=== Fuzzing Summary ===")
        print(f"Total test cases: {total}")
        print(f"Matches (semantically equivalent): {matches}")
        if total > 0:
            print(f"Semantic agreement score: {matches / total:.2%}")
        else:
            print("No valid inputs tested.")

if __name__ == "__main__":
    main()
"""

def create_fuzz_scripts(base_path: str, overwrite: bool = True):
    base = Path(base_path)
    if not base.exists():
        print(f"❌ Directory not found: {base}")
        return

    task_dirs = [d for d in base.iterdir() if d.is_dir()]
    total_folders = len(task_dirs)
    total_created = 0
    skipped_paths = []

    for task_dir in sorted(task_dirs):
        gold = task_dir / "gold.py"
        gen = task_dir / "gen.py"
        if not (gold.exists() and gen.exists()):
            skipped_paths.append(str(task_dir))
            continue

        fuzz_file = task_dir / f"diff_fuzz_{task_dir.name}.py"
        if fuzz_file.exists() and not overwrite:
            skipped_paths.append(str(task_dir))
            continue

        with open(fuzz_file, "w", encoding="utf-8") as f:
            f.write(fuzz_code_template)
        total_created += 1
        print(f"✅ Created: {fuzz_file}")

    print("\n=== Summary ===")
    print(f"Total task folders: {total_folders}")
    print(f"Total fuzz scripts created: {total_created}")
    if skipped_paths:
        print(f"⚠️ Skipped {len(skipped_paths)} folders (missing gold/gen or already exists)")


base_dataset_path = "../../../Benchmark_Curation/Diff_Fuzzing/CodeScore_Test_1000/APPS_test_0.9"
create_fuzz_scripts(base_dataset_path, overwrite=True)
