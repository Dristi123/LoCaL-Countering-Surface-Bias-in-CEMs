#!/usr/bin/env python3
import json, math
from pathlib import Path


INPUT = Path("../../../LoCaL.jsonl")  
#Update this if you have different thresholds
X_LO = 0.65  
X_HI = 0.90   
Y_LO = 0.10   
Y_HI = 0.90   


def f2(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

def in_SFD(x, y, x_hi, y_lo):  
    return (x >= x_hi) and (y <= y_lo)

def in_DFS(x, y, x_lo, y_hi):  
    return (x <= x_lo) and (y >= y_hi)

def is_mut(row):
    vt = str(row.get("variant_type", "")).lower()
    vid = str(row.get("variant_id", "")).upper()
    return ("mutation" in vt) or vid.startswith("MUT")

def is_opt(row):
    vt = str(row.get("variant_type", "")).lower()
    vid = str(row.get("variant_id", "")).upper()
    return ("optimization" in vt) or vid.startswith(("OP", "OPT"))



def pct(part, whole):
    return (100.0*part/whole) if whole > 0 else float("nan")

def main():
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}")

  
    rows = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            x = f2(row.get("surface_similarity", row.get("surfaceSim")))
            y = f2(row.get("df_score", row.get("score")))
            if math.isnan(x) or math.isnan(y):
                continue
            rows.append((x, y, is_mut(row), is_opt(row)))

  
    tot_mut = sum(1 for _,_,m,_ in rows if m)
    tot_opt = sum(1 for _,_,_,o in rows if o)

   
    base_mut_in_SFD = sum(1 for x,y,m,_ in rows if m and in_SFD(x,y,X_HI,Y_LO))
    base_opt_in_DFS = sum(1 for x,y,_,o in rows if o and in_DFS(x,y,X_LO,Y_HI))


    

    print(f"File: {INPUT}")
    print(f"Thresholds: X_LO={X_LO:.2f}, X_HI={X_HI:.2f}, Y_LO={Y_LO:.2f}, Y_HI={Y_HI:.2f}")
  
    print(f"Totals: MUT={tot_mut}  OPT={tot_opt}\n")

    print("=== Hit Rate ===")
    print(f"MUT in SFD : {base_mut_in_SFD}  ({pct(base_mut_in_SFD, tot_mut):.2f}% of MUT)")
    print(f"OPT in DFS : {base_opt_in_DFS}  ({pct(base_opt_in_DFS, tot_opt):.2f}% of OPT)\n")


if __name__ == "__main__":
    main()
