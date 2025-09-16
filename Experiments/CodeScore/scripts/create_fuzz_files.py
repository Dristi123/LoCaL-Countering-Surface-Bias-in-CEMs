import json
import argparse
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_code_file(path, code):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(code.strip() + '\n')

def create_code_dirs(data, subfolder, root_dir):
    split_dir = root_dir / subfolder
    split_dir.mkdir(parents=True, exist_ok=True)

    for entry in data:
        task_id = entry['id']
        task_dir = split_dir / task_id
        task_dir.mkdir(exist_ok=True)

        write_code_file(task_dir / 'gold.py', entry['golden_code'])
        write_code_file(task_dir / 'gen.py', entry['generated_code'])

    print(f"✅ Created {len(data)} '{subfolder}' tasks under {split_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="../data/train_2000.jsonl", help='Path to train JSONL')
    parser.add_argument('--val_file', type=str, default="../data/val_2000.jsonl", help='Path to val JSONL')
    parser.add_argument('--out_root', type=str, default='../../../Benchmark_Curation/Diff_Fuzzing', help='Output parent directory')
    args = parser.parse_args()

    train_data = load_jsonl(args.train_file)
    val_data = load_jsonl(args.val_file)

    n = len(train_data)
    base_dir = Path(args.out_root) / f"CodeScore_Base_{n}"
    base_dir.mkdir(parents=True, exist_ok=True)

    create_code_dirs(train_data, 'train', base_dir)
    create_code_dirs(val_data, 'val', base_dir)
