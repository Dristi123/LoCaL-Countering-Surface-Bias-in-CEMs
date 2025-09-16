#!/usr/bin/env bash

# 1) Build the correlation table from existing scores
python3 create_table.py

# 2) (Optional) To run inference again to get fresh metric scores
#    - Download CodeScore checkpoint: https://huggingface.co/dz1/CodeScore
#    - Place it at: ../../CodeScore/models
# python3 calculate_scores.py

# 3) Rebuild the correlation table using the fresh scores
# python3 create_table.py