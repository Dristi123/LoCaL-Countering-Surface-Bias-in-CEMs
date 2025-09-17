#!/usr/bin/env bash

# 1) Build the table from existing scores
python3 create_table.py

# 2) (Optional) To run inference again to get fresh metric scores
#    - Download CodeScore checkpoint: https://huggingface.co/dz1/CodeScore
#    - Place it at: ../../CodeScore/models
# python3 inference_on_sharecode.py
# python3 inference_on_CS.py
# python3 inference_on_LoCaL.py

# 3) Rebuild the table using the fresh scores
# python3 create_table.py
# To filter worst cases in LoCaL
#python3 filter_to_analyze.py
