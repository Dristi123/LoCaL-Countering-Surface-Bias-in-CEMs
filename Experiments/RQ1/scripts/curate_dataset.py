
import sys, json
from pathlib import Path

sys.path.insert(0, "../../Common_Scripts")
from calculate_SurfaceSim import surface_similarity

IN_JSONL = Path("../../../Source_Benchmarks/PIE/test.jsonl")
OUT_JSONL = Path("PIE_RQ1.jsonl")


with IN_JSONL.open("r", encoding="utf-8") as fin, OUT_JSONL.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rid = f"{rec.get('problem_id','')}_{rec.get('user_id','')}"
        code1 = rec.get("input","")
        code2 = rec.get("target","")
        sim = surface_similarity(code1, code2)["SurfaceSim"]
        fout.write(json.dumps({
            "id": rid,
            "golden_code": code1,
            "generated_code": code2,
            "score": 1.0,
            "surface_similarity": sim
        }, ensure_ascii=False) + "\n")
