
import json
import math
import tempfile
import subprocess
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, "../../Common_Scripts")


INPUT_JSONL   = Path("Baseline_CS.jsonl")                  
OUT_JSONL     = Path("basline_CS_with_scores.jsonl")            

LANGUAGE         = "python"
BATCH_SIZE_CBERT = 32    
LIMIT            = None   



USE_CODESCORE   = True
CODESCORE_INFER = Path("../../CodeScore/inference_orig.py")
CODESCORE_CFG = Path("../../CodeScore/configs/models/unified_metric_orig.yaml")
CODESCORE_CKPT = Path("../../CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt")

from calculate_SurfaceSim import surface_similarity


def compute_codebleu_single(ref: str, hyp: str, language: str) -> float:
    from codebleu import calc_codebleu
    try:
        res = calc_codebleu([ref], [hyp], language, weights=[0.25, 0.25, 0.25, 0.25])
        return float(res["codebleu"])
    except Exception:
        return float("nan")


def compute_crystalbleu_single(ref: str, hyp: str, language: str) -> float:
   
    from crystal_bleu_utils import crystal_BLEU
    score = crystal_BLEU([ref], [hyp], language=language)  # returns a list of scores
    
    return round(score, 2)
        


def compute_codebertscore_batch(refs, hyps, language: str, batch_size: int = 64):
   
    try:
        import code_bert_score
    except Exception:
        return [float("nan")] * len(refs)

    #print("hereeeeeeeee")
    n = len(refs)
    out = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        r_chunk = refs[start:end]
        h_chunk = hyps[start:end]
        try:
            results = code_bert_score.score(cands=h_chunk, refs=r_chunk, lang=language) 
            # print("ressssssss") 
            # print(results)
            # print([round(result.item(), 2) for result in results[2]])
            out.extend([round(result.item(), 2) for result in results[2]])
        except Exception:
            out.extend(float("nan") for _ in range(len(r_chunk)))
        
    return out


PRED_KEYS = ("predict_score", "pred_score", "predicted_score", "prediction", "score")


def compute_codescore_batch_jsonl(refs, hyps):
   
    if not USE_CODESCORE:
        return [float("nan")] * len(refs)
    if not (CODESCORE_INFER.exists() and CODESCORE_CFG.exists() and CODESCORE_CKPT.exists()):
        print("CodeScore disabled: missing inference.py, cfg, or ckpt.")
        return [float("nan")] * len(refs)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_p = td / "test.jsonl"
        out_p  = td / "out.jsonl"

     
        with test_p.open("w", encoding="utf-8") as f:
            for i, (r, h) in enumerate(zip(refs, hyps)):
                f.write(json.dumps({"id": str(i), "golden_code": r, "generated_code": h}, ensure_ascii=False) + "\n")

        cmd = [
            "python3", str(CODESCORE_INFER),
            "--cfg",       str(CODESCORE_CFG),
            "--ckpt_path", str(CODESCORE_CKPT),
            "--test_file", str(test_p),
            "--out_file",  str(out_p),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if res.returncode != 0:
                print("CodeScore inference failed:", res.stderr.strip())
                return [float("nan")] * len(refs)
        except Exception as e:
            print("CodeScore subprocess error:", e)
            return [float("nan")] * len(refs)

        id2pred = {}
        if out_p.exists():
            with out_p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    rid = str(obj.get("id", ""))
                    pred = None
                    for k in PRED_KEYS:
                        if k in obj:
                            try:
                                pred = float(obj[k])
                            except Exception:
                                pred = None
                            break
                    if rid != "" and pred is not None:
                        id2pred[rid] = pred

        return [id2pred.get(str(i), float("nan")) for i in range(len(refs))]


def load_jsonl_rows(p: Path, limit=None):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue

          
            if "golden_code" in obj and "generated_code" in obj:
                rows.append(obj)
                continue

          
            if "input" in obj and "target" in obj:
                rows.append({
                    "golden_code": obj["input"],
                    "generated_code": obj["target"],
                    **{k: v for k, v in obj.items() if k not in ("input", "target")}
                })
                continue

            
            if all(k in obj for k in ("id", "golden_code", "generated_code")):
                rows.append(obj)

    return rows


def write_jsonl_rows(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")




def compute_surfacesim(ref: str, hyp: str) -> float:
    try:
        d = surface_similarity(ref, hyp)
        return float(d.get("SurfaceSim", float("nan")))
    except Exception:
        return float("nan")


def main():
    if not INPUT_JSONL.exists():
        raise SystemExit(f"Input not found: {INPUT_JSONL}")

    data = load_jsonl_rows(INPUT_JSONL, limit=LIMIT)
    if not data:
        raise SystemExit("No rows loaded from input.")

    refs = [row["golden_code"] for row in data]
    hyps = [row["generated_code"] for row in data]
    
  
    cbert = compute_codebertscore_batch(refs, hyps, LANGUAGE, BATCH_SIZE_CBERT)
    cscores = compute_codescore_batch_jsonl(refs, hyps)

    out_rows = []
    for i, row in enumerate(data):
        ref = refs[i]
        hyp = hyps[i]

        cbleu = compute_codebleu_single(ref, hyp, LANGUAGE)
        crys  = compute_crystalbleu_single(ref, hyp, LANGUAGE)
        cbert_i = cbert[i] if i < len(cbert) else float("nan")
        cscore  = cscores[i] if i < len(cscores) else float("nan")

      
        ss = compute_surfacesim(ref, hyp)
        gt = row.get("score", None)  
        if gt is None or math.isnan(ss):
            abs_diff = float("nan")
        else:
            try:
                abs_diff = abs(float(gt) - ss)
            except Exception:
                abs_diff = float("nan")

        row["codebleu"]      = None if math.isnan(cbleu)   else cbleu
        row["crystalbleu"]   = None if math.isnan(crys)    else crys
        row["codebertscore"] = None if math.isnan(cbert_i) else cbert_i
        row["codescore"]     = None if math.isnan(cscore)  else cscore
        row["surfaceSim"]    = None if math.isnan(ss)      else ss
        row["abs_surfaceSim_minus_score"] = None if math.isnan(abs_diff) else abs_diff

        out_rows.append(row)

    write_jsonl_rows(OUT_JSONL, out_rows)
    print(f"Output stored in {OUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
