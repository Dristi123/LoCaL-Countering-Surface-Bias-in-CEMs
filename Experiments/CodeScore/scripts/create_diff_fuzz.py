import os
from pathlib import Path

# Actual Python code template (not as a string assigned to a variable)
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

def create_fuzz_scripts(base_path):
    total_created = 0
    total_folders = 0
    skipped_paths = []

    for split in ["train", "val"]:
        split_dir = Path(base_path) / split
        if not split_dir.exists():
            print(f"Directory not found: {split_dir}")
            continue

        for task_dir in split_dir.iterdir():
            if task_dir.is_dir():
                total_folders += 1
                gold = task_dir / "gold.py"
                gen = task_dir / "gen.py"

                if not (gold.exists() and gen.exists()):
                    skipped_paths.append(str(task_dir))
                    continue

                fuzz_file = task_dir / f"diff_fuzz_{task_dir.name}.py"
                with open(fuzz_file, 'w', encoding='utf-8') as f:
                    f.write(fuzz_code_template)
                total_created += 1
                print(f"✅ Created: {fuzz_file}")

    print("\n=== Summary ===")
    print(f"Total folders: {total_folders}")
    print(f"Total fuzz scripts created: {total_created}")
    print(f"⚠️ Skipped: {len(skipped_paths)} folders (missing gold.py or gen.py)")

    if skipped_paths:
        print("\n📁 Skipped folders:")
        for path in skipped_paths:
            print(f" - {path}")


base_dataset_path = "../../../Benchmark_Curation/Diff_Fuzzing/CodeScore_Base_2000"
create_fuzz_scripts(base_dataset_path)
