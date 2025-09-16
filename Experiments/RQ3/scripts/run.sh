#!/usr/bin/env bash


## Calculate threshold and plot graph from existing sampling
THRESH_SCRIPT="calculate_thresholds.py"   
PLOT_SCRIPT="plot_fig.py"     
## (Optional) Uncomment this to redo sampling
#python3 split_data.py
OUT="$(python3 "$THRESH_SCRIPT")"
echo "$OUT"

LINE="$(printf "%s\n" "$OUT" | grep -E 'τx_lo|x_lo' | head -n1)"
read TX_LO TX_HI TY_LO TY_HI < <(printf "%s" "$LINE" | grep -oE '[0-9]+(\.[0-9]+)?' | head -n4 | tr '\n' ' ')

echo "Using thresholds: tx_lo=$TX_LO tx_hi=$TX_HI ty_lo=$TY_LO ty_hi=$TY_HI"
python3 "$PLOT_SCRIPT" --tx-lo "$TX_LO" --tx-hi "$TX_HI" --ty-lo "$TY_LO" --ty-hi "$TY_HI"


