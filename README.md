# LoCaL-Countering-Surface-Bias-in-CEMs
This repository contains the replication package that generates results shown in the submission.
<br>**For consistency, figures and tables in this README use the same numbering as the paper.</br>
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
![Table 3 — RQ1 results](assets/table_3.png)

## Reproducing RQ2 results
```bash
cd Experiments/RQ2/scripts
chmod +x run.sh
./run.sh
```
This will generate Table 4 inside the [`Experiments/RQ2/results`](Experiments/RQ2/results) directory.
![Table 4 — RQ2 results](assets/table_4.png)

## Reproducing RQ3 results
```bash
cd Experiments/RQ3/scripts
chmod +x run.sh
./run.sh
```
This will generate Figure 5 inside the [`Experiments/RQ3/results`](Experiments/RQ3/results) directory.
![Figure 5 — RQ3_results](assets/scatter_plot.png)

## Reproducing RQ4 results
```bash
cd Experiments/RQ4/scripts
chmod +x run.sh
./run.sh
```
This will generate Table 5 inside the [`Experiments/RQ4/results`](Experiments/RQ4/results) directory.
![Table 5 — RQ4 results](assets/table_5.png)

## Reproducing RQ5 results
1. Navigate to the scripts directory
```bash
cd Experiments/RQ5/scripts
```
2. Download trained models from [trained models](https://zenodo.org/records/17139161?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVlOTg3NjkzLTEwYzktNDYyMy1hNDQzLTA4ZmJkYTEwMjkzZiIsImRhdGEiOnt9LCJyYW5kb20iOiJjZjk5YTc0NTE1M2U0MDkyZGRkNWUwNmNiNmJjMzEyMiJ9.ybNeG0zu_Wv32PHgHQtwLl_HCuVevXiRwOS_ewEsFcXDNbxKUxaZFKK0hcePTxUww3Yc4eUTi8vXTSC026R93g)
3. Follow the instructions in run.sh to generate results
```bash
chmod +x run.sh
./run.sh
```
4. (Optional) To retrain the models 
  - Download the training and validation splits for each p(percentage of LoCaL) from [data_for_RQ5](https://zenodo.org/records/17139161?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImVlOTg3NjkzLTEwYzktNDYyMy1hNDQzLTA4ZmJkYTEwMjkzZiIsImRhdGEiOnt9LCJyYW5kb20iOiJjZjk5YTc0NTE1M2U0MDkyZGRkNWUwNmNiNmJjMzEyMiJ9.ybNeG0zu_Wv32PHgHQtwLl_HCuVevXiRwOS_ewEsFcXDNbxKUxaZFKK0hcePTxUww3Yc4eUTi8vXTSC026R93g)
  - Refer to the [CodeScore repository](https://github.com/Dingjz/CodeScore) for training instructions.
This will generate Table 6 inside the [`Experiments/RQ5/results`](Experiments/RQ5/results) directory.
![Table 6 — RQ5 results](assets/table_6.png)

## Literature Survey
``` bash
python3 get_stats.py
```
