#!/usr/bin/env python3
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

def load_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def write_code_file(path: Path, code: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code.strip() + '\n')

def rename_ids_with_occurrences(data):
    """
    In-place: make each entry['id'] unique by appending _<occurrence> per original order.
    Example: test-1772, test-1772, test-1772 -> test-1772_1, test-1772_2, test-1772_3
    """
    counts = defaultdict(int)
    for entry in data:
        base_id = entry['id']
        counts[base_id] += 1
        entry['id'] = f"{base_id}_{counts[base_id]}"

def create_code_dirs(data, subfolder: str, root_dir: Path):
    split_dir = root_dir / subfolder
    split_dir.mkdir(parents=True, exist_ok=True)

    for entry in data:
        task_id = entry['id']  # already unique (id_<occ>)
        task_dir = split_dir / task_id
        task_dir.mkdir(exist_ok=True)

        write_code_file(task_dir / 'gold.py', entry['golden_code'])
        write_code_file(task_dir / 'gen.py', entry['generated_code'])

    print(f"✅ Created {len(data)} '{subfolder}' tasks under {split_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=str, default="../data/APPS_test_0.9.jsonl", help='Path to JSONL')
    parser.add_argument('--out_root', type=str, default='../../../Benchmark_Curation/Diff_Fuzzing', help='Output parent directory')
    parser.add_argument('--sample_n', type=int, default=1000, help='Random sample size (use all if fewer)')
    parser.add_argument('--seed', type=int, default=13, help='Random seed for reproducibility')
    args = parser.parse_args()

    in_path = Path(args.jsonl)
    out_root = Path(args.out_root)
    stem = in_path.stem

    # 1) Load and rename IDs with occurrence indices
    full = load_jsonl(str(in_path))
    rename_ids_with_occurrences(full)

    # Base dir named by sampled size (filled after sampling)
    # But save the fully-renamed JSONL first for traceability.
    renamed_dir = out_root / "renamed_jsonl"
    renamed_path = renamed_dir / f"{stem}_renamed.jsonl"
    save_jsonl(full, renamed_path)
    print(f"💾 Renamed JSONL saved to: {renamed_path}")

    # 2) Sample (after renaming, so IDs remain traceable to original occurrence)
    random.seed(args.seed)
    if len(full) > args.sample_n:
        data = random.sample(full, args.sample_n)
    else:
        data = full

    n = len(data)
    base_dir = out_root / f"CodeScore_Test_{n}"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Also save the sampled subset JSONL for exact reproducibility of folders
    sampled_path = base_dir / f"{stem}_sampled.jsonl"
    save_jsonl(data, sampled_path)
    print(f"💾 Sampled subset ({n}) saved to: {sampled_path}")

    # 3) Create per-task folders/files
    create_code_dirs(data, stem, base_dir)

if __name__ == '__main__':
    main()
