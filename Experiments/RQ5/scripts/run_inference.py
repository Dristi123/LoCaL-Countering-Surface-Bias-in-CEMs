#!/usr/bin/env python3
from pathlib import Path
import subprocess
from datetime import datetime
import re
import sys

CKPT_ROOT = Path("../CodeScore/CodeScore/all.uni")
CODESCORE = Path("../CodeScore/CodeScore")
INFER_PY  = CODESCORE / "inference.py"
CFG_FILE  = CODESCORE / "configs/models/unified_metric.yaml"


TEST_FILES = {
    "local": Path("local_test.jsonl"),
    "baseline_sampled": Path("Baseline.jsonl"),
}

RUN_TAG   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT  = Path(f"cs_infer_results_{RUN_TAG}")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    i = 1
    while True:
        q = p.with_name(f"{p.stem}__{i}{p.suffix}")
        if not q.exists():
            return q
        i += 1

def slug(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", s)
    return s.strip("_") or "unnamed"

def die(msg: str, code: int = 1):
    print(f"[error] {msg}", file=sys.stderr)
    sys.exit(code)

if not INFER_PY.is_file():
    die(f"inference.py not found at {INFER_PY}")
if not CFG_FILE.is_file():
    die(f"config file not found at {CFG_FILE}")

ckpts = sorted(CKPT_ROOT.rglob("checkpoints/*.ckpt"))
if not ckpts:
    die(f"no checkpoints found under {CKPT_ROOT}/**/checkpoints/*.ckpt")

active_tests = {}
for label, path in TEST_FILES.items():
    if path.is_file():
        active_tests[slug(label)] = path
    else:
        print(f"[warn] missing test file: {path} (label={label})")

if not active_tests:
    die("no valid test files to run")

print(f"[info] using {len(ckpts)} checkpoints")
print(f"[info] test sets: {', '.join(active_tests.keys())}")
print(f"[info] outputs under: {OUT_ROOT.resolve()}")

for ckpt in ckpts:
    exp_name = ckpt.parent.parent.name
    for label, test_path in active_tests.items():
        exp_dir = OUT_ROOT / exp_name / label
        exp_dir.mkdir(parents=True, exist_ok=True)

        out_path = unique_path(exp_dir / f"{ckpt.stem}.jsonl")
        cmd = [
            "python", str(INFER_PY),
            "--cfg", str(CFG_FILE),
            "--ckpt_path", str(ckpt),
            "--test_file", str(test_path),
            "--out_file", str(out_path),
        ]
        print(f"[run] {ckpt.name} | test={label} -> {out_path}")
        subprocess.run(cmd, check=False)

print(f"all outputs under: {OUT_ROOT.resolve()}")
