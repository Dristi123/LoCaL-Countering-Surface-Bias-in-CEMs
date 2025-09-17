#!/usr/bin/env bash

# 1) Build the table from existing scores
python3 create_table.py

# 2) To rerun inference 
#    - Download all checkpoints from: https://huggingface.co/dz1/CodeScore
#    - Place it at: ../../CodeScore/all.uni/
# python3 run_inference.py
# python3 calculate MAE

# 3) Rebuild the table 
# python3 create_table.py