#!/usr/bin/env bash

# 1) Build the table from existing scores
python3 create_table.py

# 2) To rerun inference 
#    - Download all checkpoints from: https://zenodo.org/records/17139161?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVlOTg3NjkzLTEwYzktNDYyMy1hNDQzLTA4ZmJkYTEwMjkzZiIsImRhdGEiOnt9LCJyYW5kb20iOiJjZjk5YTc0NTE1M2U0MDkyZGRkNWUwNmNiNmJjMzEyMiJ9.ybNeG0zu_Wv32PHgHQtwLl_HCuVevXiRwOS_ewEsFcXDNbxKUxaZFKK0hcePTxUww3Yc4eUTi8vXTSC026R93g
#    - Place it at: ../../CodeScore/all.uni
# python3 run_inference.py
# python3 calculate MAE

# 3) Rebuild the table 
# python3 create_table.py