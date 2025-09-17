# LoCaL-Countering-Surface-Bias-in-CEMs
This repository contains the replication package that generates results shown in the submission.
## Conda Environment
1. Create a Conda env
```bash
conda create -n local python=3.8.20
```
2. Activate the environment
```bash
conda activate local
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
## Reproduce RQ1 results
```bash
cd Experiments/RQ1/scripts
chmod +x run.sh
./run.sh
```
This will generate Table 3 inside the [`Experiments/RQ1/results`](Experiments/RQ1/results) directory.
