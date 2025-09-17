
from pathlib import Path
import subprocess
from datetime import datetime
import re
import sys

CKPT_ROOT = Path("../../CodeScore/all.uni")
CODESCORE = Path("../../CodeScore")
INFER_PY  = CODESCORE / "inference.py"
CFG_FILE  = CODESCORE / "configs/models/unified_metric.yaml"

TEST_FILES = {
    "local": Path("Local_test.jsonl"),
    "baseline": Path("Baseline.jsonl"),
}

RUN_TAG  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_ROOT = Path(f"cs_infer_results_{RUN_TAG}")
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

if not CKPT_ROOT.is_dir():
    die(f"checkpoint root not found: {CKPT_ROOT}")
if not INFER_PY.is_file():
    die(f"inference.py not found at {INFER_PY}")
if not CFG_FILE.is_file():
    die(f"config file not found at {CFG_FILE}")


ckpts = sorted(CKPT_ROOT.glob("**/*.ckpt"))
if not ckpts:
    die(f"no checkpoints found under {CKPT_ROOT}/**/*.ckpt")

active_tests = {}
for label, path in TEST_FILES.items():
    if path.is_file():
        active_tests[slug(label)] = path
    else:
        print(f"[warn] missing test file: {path} (label={label})")

if not active_tests:
    die("no valid test files to run")

print(f"Using {len(ckpts)} checkpoints")
print(f"Test sets: {', '.join(active_tests.keys())}")
print(f"Outputs stored in {OUT_ROOT.resolve()}")

PY = sys.executable

for ckpt in ckpts:
    try:
        rel_parent = ckpt.parent.relative_to(CKPT_ROOT).as_posix()  
    except ValueError:
        rel_parent = ckpt.parent.name
    exp_name = slug(rel_parent.replace("/", "__"))

    for label, test_path in active_tests.items():
        exp_dir = OUT_ROOT / exp_name / label
        exp_dir.mkdir(parents=True, exist_ok=True)
        out_path = unique_path(exp_dir / f"{ckpt.stem}.jsonl")
        cmd = [
            PY, str(INFER_PY),
            "--cfg", str(CFG_FILE),
            "--ckpt_path", str(ckpt),
            "--test_file", str(test_path),
            "--out_file", str(out_path),
        ]
        subprocess.run(cmd, check=False)

print(f"all outputs under: {OUT_ROOT.resolve()}")
