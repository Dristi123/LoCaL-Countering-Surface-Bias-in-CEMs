#!/usr/bin/env python3
import json, math
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,        
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 14,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
})


INPUT = Path("combined_dataset.jsonl")
SURF, SEM = "surfaceSim", "score"


TX_LO = 0.65
TX_HI = 0.90
TY_LO = 0.10
TY_HI = 0.90

OUT_DIR  = Path("../results")
OUT_STEM = "scatter_plot"   


POINT_SIZE  = 26
ALPHA       = 1
YLIM_TOP    = 1.02
XLIM_RIGHT  = 1.01   

FIG_SIZE = (9.6, 5.8)


COLOR_BASELINE = "#1f77b4"  
COLOR_LOCAL    = "#d62728"  
COLOR_OTHERS   = "#9e9e9e"  

DFS_FILL_RGBA = (0.66, 0.87, 0.68, 0.40)  
SFD_FILL_RGBA = (0.86, 0.88, 0.61, 0.70)  


def f2(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except:
        return float("nan")

def infer_source(obj):
    rid = str(obj.get("id", ""))
    return "local" if rid.startswith("LOCAL") else "baseline"

def load_points(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            obj = json.loads(s)
            x = f2(obj.get(SURF)); y = f2(obj.get(SEM))
            if math.isnan(x) or math.isnan(y): continue
            rows.append((x, y, infer_source(obj)))
    return rows

def apply_axis_style(ax):
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_edgecolor("black")
    ax.tick_params(direction="out", length=4, width=1)

def draw_left_xaxis_break(ax, x_left, frac=0.015):
    """Two small slanted '//' marks at the left x-axis to indicate truncation."""
    trans = ax.get_xaxis_transform()
    xr = ax.get_xlim()
    span = xr[1] - xr[0]
    dx = span * frac
    y_up, y_dn = 0.02, -0.02
    ax.plot([x_left + 0.002*span, x_left + 0.002*span + dx],
            [y_dn, y_up], transform=trans, color="black", clip_on=False, linewidth=1.2)
    ax.plot([x_left + 0.004*span, x_left + 0.004*span + dx],
            [y_dn, y_up], transform=trans, color="black", clip_on=False, linewidth=1.2)

def round_up_to_step(x, step=0.2):
    import math
    return round(math.ceil(x/step)*step, 10)

def main():
    if not INPUT.exists():
        raise SystemExit(f"Input not found: {INPUT}")

    pts = load_points(INPUT)
    if not pts:
        raise SystemExit("No valid rows in input.")

 
    bx = [x for x, y, s in pts if s == "baseline"]
    by = [y for x, y, s in pts if s == "baseline"]
    lx = [x for x, y, s in pts if s == "local"]
    ly = [y for x, y, s in pts if s == "local"]
    ox = [x for x, y, s in pts if s not in ("baseline", "local")]
    oy = [y for x, y, s in pts if s not in ("baseline", "local")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIG_SIZE)

  
    dfs_w, dfs_h = max(TX_LO - 0.0, 0), max(YLIM_TOP - TY_HI, 0)
    ax.add_patch(Rectangle((0.0, TY_HI), width=dfs_w, height=dfs_h,
                           facecolor=DFS_FILL_RGBA, edgecolor='none', zorder=0))
  
    sfd_w, sfd_h = max(XLIM_RIGHT - TX_HI, 0), max(TY_LO - 0.0, 0)
    ax.add_patch(Rectangle((TX_HI, 0.0), width=sfd_w, height=sfd_h,
                           facecolor=SFD_FILL_RGBA, edgecolor='none', zorder=0))

  
    if bx:
        ax.scatter(bx, by, s=POINT_SIZE, alpha=ALPHA, c=COLOR_BASELINE,
                   edgecolors="none", label=r"$\mathrm{CodeScore}_{\mathrm{test}}$", zorder=2)
    if lx:
        ax.scatter(lx, ly, s=POINT_SIZE, alpha=ALPHA, c=COLOR_LOCAL,
                   edgecolors="none", label="LoCaL", zorder=3)
    if ox:
        ax.scatter(ox, oy, s=POINT_SIZE, alpha=0.45, c=COLOR_OTHERS,
                   edgecolors="none", label="_nolegend_", zorder=1)

   
    all_x = [*bx, *lx, *ox] if (bx or lx or ox) else [0.0]
    min_x = min(all_x)
    x_left = max(0.0, min(TX_LO - 0.05, min_x - 0.02)) 
    ax.set_xlim(x_left, XLIM_RIGHT)
    ax.set_ylim(0.0, YLIM_TOP)

    if x_left > 0.0:
        draw_left_xaxis_break(ax, x_left)

    # Threshold lines
    ax.axvline(TX_LO, linestyle="--", linewidth=1.8, color="black", zorder=4)
    ax.axvline(TX_HI, linestyle="--", linewidth=1.8, color="black", zorder=4)
    ax.axhline(TY_LO, linestyle="--", linewidth=1.8, color="black", zorder=4)
    ax.axhline(TY_HI, linestyle="--", linewidth=1.8, color="black", zorder=4)

   
    start_x = round_up_to_step(ax.get_xlim()[0], 0.2)
    xticks = [round(start_x + i*0.2, 2) for i in range(int((1.0 - start_x)/0.2) + 1)]
    ax.set_xticks(xticks)

    xlabels = []
    for t in xticks:
        if abs(t - TX_LO) < 1e-6:
            xlabels.append("")  # custom label added below
        elif abs(t - TX_HI) < 1e-6:
            xlabels.append(rf"$x_{{\mathrm{{hi}}}}={t:.2f}$")
        else:
            xlabels.append(f"{t:.2f}")
    ax.set_xticklabels(xlabels)

   
    ax.text(TX_LO + 0.09, -0.016, rf"$x_{{\mathrm{{lo}}}}={TX_LO:.2f}$",
            transform=ax.get_xaxis_transform(), ha='right', va='top', color='black')
    ax.text(TX_HI - 0.04, -0.016, rf"$x_{{\mathrm{{hi}}}}={TX_HI:.2f}$",
        transform=ax.get_xaxis_transform(), ha='left', va='top', color='black')

  
    yticks = [round(i*0.2, 2) for i in range(0, 6)]
    ax.set_yticks(yticks)
    ylabels = []
    for t in yticks:
        if abs(t - TY_LO) < 1e-6:
            ylabels.append(rf"$y_{{\mathrm{{lo}}}}={t:.2f}$")
        elif abs(t - TY_HI) < 1e-6:
            ylabels.append(rf"$y_{{\mathrm{{hi}}}}={t:.2f}$")
        else:
            ylabels.append(f"{t:.2f}")
    ax.set_yticklabels(ylabels)
    ax.text(-0.02, TY_LO, rf"$y_{{\mathrm{{lo}}}}={TY_LO:.2f}$",
        transform=ax.get_yaxis_transform(), ha='right', va='center', color='black')

    ax.text(-0.02, TY_HI, rf"$y_{{\mathrm{{hi}}}}={TY_HI:.2f}$",
        transform=ax.get_yaxis_transform(), ha='right', va='center', color='black')
 
    ax.set_xlabel("SurfaceSim")
    ax.set_ylabel(r"$\mathrm{df}_{\mathrm{score}}$")

    handles = []
    if bx:
        handles.append(Line2D([0],[0], marker="o", color="none",
                              markerfacecolor=COLOR_BASELINE, markersize=8,
                              label=r"$\mathrm{CodeScore}_{\mathrm{test}}$"))
    if lx:
        handles.append(Line2D([0],[0], marker="o", color="none",
                              markerfacecolor=COLOR_LOCAL, markersize=8, label="LoCaL"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 1.015), frameon=False)

    apply_axis_style(ax)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{OUT_STEM}.png"
    pdf_path = OUT_DIR / f"{OUT_STEM}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path,            bbox_inches="tight")
    plt.close(fig)
    print(f" saved {png_path.resolve()}")
    print(f"saved  {pdf_path.resolve()}")
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx-lo", type=float)
    ap.add_argument("--tx-hi", type=float)
    ap.add_argument("--ty-lo", type=float)
    ap.add_argument("--ty-hi", type=float)
    args = ap.parse_args()
    TX_LO, TX_HI, TY_LO, TY_HI = args.tx_lo, args.tx_hi, args.ty_lo, args.ty_hi
    main()
