#!/usr/bin/env bash

# 1) Build the distinguishability table from existing runs
python3 create_table.py

# 2) (Optional) To redo sampling
#    - Download CodeScore checkpoint: https://huggingface.co/dz1/CodeScore
#    - Place it at: ../../CodeScore/models
# python3 analyze_distinguishability.py

# 3) Rebuild the distinguishability table from new runs
# python3 create_table.py