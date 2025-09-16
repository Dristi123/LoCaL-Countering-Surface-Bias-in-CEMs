#!/usr/bin/env python3
import json, math
from pathlib import Path

# --------- config ----------
INPUT = Path("../../../Common_Scripts/LoCaL_with_PIEextra_mutOnly.jsonl")   # your LoCaL JSONL
X_LO = 0.65   # DFS:  x <= X_LO
X_HI = 0.90   # SFD:  x >= X_HI
Y_LO = 0.10   # SFD:  y <= Y_LO
Y_HI = 0.90   # DFS:  y >= Y_HI

AREA_GROWTHS = [0.05, 0.20]  # +5%, +10% area expansions
# --------------------------

def f2(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

def in_SFD(x, y, x_hi, y_lo):  # bottom-right
    return (x >= x_hi) and (y <= y_lo)

def in_DFS(x, y, x_lo, y_hi):  # top-left
    return (x <= x_lo) and (y >= y_hi)

def is_mut(row):
    vt = str(row.get("variant_type", "")).lower()
    vid = str(row.get("variant_id", "")).upper()
    return ("mutation" in vt) or vid.startswith("MUT")

def is_opt(row):
    vt = str(row.get("variant_type", "")).lower()
    vid = str(row.get("variant_id", "")).upper()
    return ("optimization" in vt) or vid.startswith(("OP", "OPT"))

def expand_dfs(x_lo, y_hi, alpha):
    """Increase DFS area by factor (1+alpha) via side scaling."""
    s = math.sqrt(1.0 + alpha)
    x_lo2 = min(1.0, x_lo * s)
    y_hi2 = 1.0 - min(1.0, (1.0 - y_hi) * s)
    return x_lo2, y_hi2

def expand_sfd(x_hi, y_lo, alpha):
    """Increase SFD area by factor (1+alpha) via side scaling."""
    s = math.sqrt(1.0 + alpha)
    width2 = min(1.0, (1.0 - x_hi) * s)
    x_hi2 = 1.0 - width2
    y_lo2 = min(1.0, y_lo * s)
    return x_hi2, y_lo2

def pct(part, whole):
    return (100.0*part/whole) if whole > 0 else float("nan")

def main():
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}")

    # Accumulate rows
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

    # Totals by type
    tot_mut = sum(1 for _,_,m,_ in rows if m)
    tot_opt = sum(1 for _,_,_,o in rows if o)

    # Base counts
    base_mut_in_SFD = sum(1 for x,y,m,_ in rows if m and in_SFD(x,y,X_HI,Y_LO))
    base_opt_in_DFS = sum(1 for x,y,_,o in rows if o and in_DFS(x,y,X_LO,Y_HI))

    # Areas (for info)
    area_DFS = X_LO * (1.0 - Y_HI)
    area_SFD = (1.0 - X_HI) * Y_LO

    print(f"File: {INPUT}")
    print(f"Base thresholds: X_LO={X_LO:.2f}, X_HI={X_HI:.2f}, Y_LO={Y_LO:.2f}, Y_HI={Y_HI:.2f}")
    print(f"Base areas: DFS={area_DFS:.4f}  SFD={area_SFD:.4f}")
    print(f"Totals: MUT={tot_mut}  OPT={tot_opt}\n")

    print("=== Base coverage ===")
    print(f"MUT in SFD : {base_mut_in_SFD}  ({pct(base_mut_in_SFD, tot_mut):.2f}% of MUT)")
    print(f"OPT in DFS : {base_opt_in_DFS}  ({pct(base_opt_in_DFS, tot_opt):.2f}% of OPT)\n")

    # Expanded regions (+5%, +10% area)
    for alpha in AREA_GROWTHS:
        x_lo2, y_hi2 = expand_dfs(X_LO, Y_HI, alpha)
        x_hi2, y_lo2 = expand_sfd(X_HI, Y_LO, alpha)

        mut_in_SFD_2 = sum(1 for x,y,m,_ in rows if m and in_SFD(x,y,x_hi2,y_lo2))
        opt_in_DFS_2 = sum(1 for x,y,_,o in rows if o and in_DFS(x,y,x_lo2,y_hi2))

        delta_mut = mut_in_SFD_2 - base_mut_in_SFD
        delta_opt = opt_in_DFS_2 - base_opt_in_DFS

        area_DFS2 = x_lo2 * (1.0 - y_hi2)
        area_SFD2 = (1.0 - x_hi2) * y_lo2

        print(f"=== +{int(alpha*100)}% area expansion ===")
        print(f"DFS thr → x_lo={x_lo2:.3f}, y_hi={y_hi2:.3f}  | area {area_DFS2:.4f}")
        print(f"SFD thr → x_hi={x_hi2:.3f}, y_lo={y_lo2:.3f}  | area {area_SFD2:.4f}")
        print(f"MUT in SFD : {mut_in_SFD_2}  (+{delta_mut})  ({pct(mut_in_SFD_2, tot_mut):.2f}% of MUT)")
        print(f"OPT in DFS : {opt_in_DFS_2}  (+{delta_opt})  ({pct(opt_in_DFS_2, tot_opt):.2f}% of OPT)")
        print()

if __name__ == "__main__":
    main()
