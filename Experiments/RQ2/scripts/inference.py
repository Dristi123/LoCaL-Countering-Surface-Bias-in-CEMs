
import sys, json, math, subprocess, tempfile
from pathlib import Path

DEFAULT_LANGUAGE = "python"
DEFAULT_BATCH_CBERT = 32

CODESCORE_INFER = Path("../../CodeScore/inference_orig.py")
CODESCORE_CFG = Path("../../CodeScore/configs/models/unified_metric_orig.yaml")
CODESCORE_CKPT = Path("../../CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt")

sys.path.insert(0, "../../Common_Scripts")
PRED_KEYS = ("predict_score", "pred_score", "predicted_score", "prediction", "score")

def load_jsonl_rows(p, limit=None):
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
                rows.append(obj); continue
            if "input" in obj and "target" in obj:
                rows.append({"golden_code": obj["input"], "generated_code": obj["target"], **{k: v for k, v in obj.items() if k not in ("input","target")}}); continue
            if all(k in obj for k in ("id","golden_code","generated_code")):
                rows.append(obj)
    return rows

def write_jsonl_rows(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def compute_codebleu_single(ref, hyp, language):
    try:
        from codebleu import calc_codebleu
        res = calc_codebleu([ref],[hyp], language, weights=[0.25,0.25,0.25,0.25])
        return float(res.get("codebleu", float("nan")))
    except Exception:
        return float("nan")

def compute_crystalbleu_single(ref, hyp, language):
    try:
        from crystal_bleu_utils import crystal_BLEU
        score = crystal_BLEU([ref],[hyp], language=language)
        if isinstance(score,(list,tuple)):
            score = score[0] if score else float("nan")
        return float(score)
    except Exception:
        return float("nan")

def compute_codebertscore_batch(refs, hyps, language, batch_size=64):
    try:
        import code_bert_score
    except Exception:
        return [float("nan")]*len(refs)
    n = len(refs)
    out = []
    for start in range(0,n,batch_size):
        end = min(start+batch_size,n)
        r_chunk = refs[start:end]
        h_chunk = hyps[start:end]
        try:
            results = code_bert_score.score(cands=h_chunk, refs=r_chunk, lang=language)
            out.extend([round(result.item(),2) for result in results[2]])
        except Exception:
            out.extend(float("nan") for _ in range(len(r_chunk)))
    return out

def compute_codescore_batch_jsonl(refs, hyps, infer_py, cfg, ckpt, enabled=True):
    if not enabled:
        return [float("nan")]*len(refs)
    if not (infer_py.exists() and cfg.exists() and ckpt.exists()):
        return [float("nan")]*len(refs)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_p = td/"test.jsonl"
        out_p = td/"out.jsonl"
        with test_p.open("w", encoding="utf-8") as f:
            for i,(r,h) in enumerate(zip(refs,hyps)):
                f.write(json.dumps({"id":str(i),"golden_code":r,"generated_code":h}, ensure_ascii=False)+"\n")
        cmd = ["python3", str(infer_py), "--cfg", str(cfg), "--ckpt_path", str(ckpt), "--test_file", str(test_p), "--out_file", str(out_p)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if res.returncode != 0:
                return [float("nan")]*len(refs)
        except Exception:
            return [float("nan")]*len(refs)
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
                    rid = str(obj.get("id",""))
                    pred = None
                    for k in PRED_KEYS:
                        if k in obj:
                            try:
                                pred = float(obj[k])
                            except Exception:
                                pred = None
                            break
                    if rid and pred is not None:
                        id2pred[rid] = pred
        return [id2pred.get(str(i), float("nan")) for i in range(len(refs))]

try:
    from calculate_SurfaceSim import surface_similarity as _surface_similarity
except Exception:
    _surface_similarity = None

def compute_surfacesim(ref, hyp):
    if _surface_similarity is None:
        return float("nan")
    try:
        d = _surface_similarity(ref, hyp)
        if isinstance(d, dict):
            return float(d.get("SurfaceSim", float("nan")))
        return float(d)
    except Exception:
        return float("nan")

def _get_arg(flag, default=None):
    a = sys.argv
    if flag in a:
        i = a.index(flag)
        if i+1 < len(a):
            return a[i+1]
    return default

def _has(flag):
    return flag in sys.argv

def main():
    in_path = Path(_get_arg("--in", "in.jsonl"))
    out_path = Path(_get_arg("--out", "out.jsonl"))
    language = _get_arg("--language", DEFAULT_LANGUAGE)
    try:
        batch = int(_get_arg("--batch-size-cbert", str(DEFAULT_BATCH_CBERT)))
    except Exception:
        batch = DEFAULT_BATCH_CBERT
    try:
        limit_val = _get_arg("--limit", None)
        limit = int(limit_val) if limit_val is not None else None
    except Exception:
        limit = None
    skip_codescore = _has("--skip-codescore")
    infer_py = Path(_get_arg("--codescore-infer", str(CODESCORE_INFER)))
    cfg = Path(_get_arg("--codescore-cfg", str(CODESCORE_CFG)))
    ckpt = Path(_get_arg("--codescore-ckpt", str(CODESCORE_CKPT)))
    if not in_path.exists():
        raise SystemExit(f"[error] input not found: {in_path}")
    data = load_jsonl_rows(in_path, limit=limit)
    if not data:
        raise SystemExit("[error] no rows loaded from input.")
    refs = [row["golden_code"] for row in data]
    hyps = [row["generated_code"] for row in data]
    cbert = compute_codebertscore_batch(refs, hyps, language, batch)
    cscores = compute_codescore_batch_jsonl(refs, hyps, infer_py, cfg, ckpt, enabled=(not skip_codescore))
    out_rows = []
    for i,row in enumerate(data):
        ref = refs[i]; hyp = hyps[i]
        cbleu = compute_codebleu_single(ref, hyp, language)
        crys = compute_crystalbleu_single(ref, hyp, language)
        cbert_i = cbert[i] if i < len(cbert) else float("nan")
        cscore = cscores[i] if i < len(cscores) else float("nan")
        ss = compute_surfacesim(ref, hyp)
        gt = row.get("score", None)
        if gt is None or math.isnan(ss):
            abs_diff = float("nan")
        else:
            try:
                abs_diff = abs(float(gt) - ss)
            except Exception:
                abs_diff = float("nan")
        row["codebleu"] = None if math.isnan(cbleu) else cbleu
        row["crystalbleu"] = None if math.isnan(crys) else crys
        row["codebertscore"] = None if math.isnan(cbert_i) else cbert_i
        row["codescore"] = None if math.isnan(cscore) else cscore
        row["surfaceSim"] = None if math.isnan(ss) else ss
        row["abs_surfaceSim_minus_score"] = None if math.isnan(abs_diff) else abs_diff
        out_rows.append(row)
    write_jsonl_rows(out_path, out_rows)
    print(str(out_path.resolve()))

if __name__ == "__main__":
    main()
