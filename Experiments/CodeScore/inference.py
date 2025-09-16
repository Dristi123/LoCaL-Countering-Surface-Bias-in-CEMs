#!/usr/bin/env python3

# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python inference.py --cfg configs/models/unified_metric.yaml --ckpt_path models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt --test_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/tests/temp_data.jsonl --out_file tests/
Command for training new Metrics.
=================================

e.g:
```
    comet-train --cfg configs/models/regression_metric.yaml --seed_everything 12
```

For more details run the following command:
```
    comet-train --help
```
"""
import json
import logging
import warnings
import sys, os
import torch
from jsonargparse import ActionConfigFile, ArgumentParser, namespace_to_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger

from models import (
    RankingMetric,
    ReferencelessRegression,
    RegressionMetric,
    UnifiedMetric,
)
#from models import UnifiedMetric_exec as UnifiedMetric
from utils import write_jsonl

logger = logging.getLogger(__name__)
# import wandb
# wandb.finish()
# wandb_logger = WandbLogger(project='reg_pass@k.n')



def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command for training COMET models.")
    parser.add_argument(
        "--seed_everything",
        type=int,
        default=12,
        help="Training Seed.",
    )
    parser.add_argument("--cfg", action=ActionConfigFile)
    parser.add_argument("--test_file", type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_subclass_arguments(RegressionMetric, "regression_metric")
    parser.add_subclass_arguments(
        ReferencelessRegression, "referenceless_regression_metric"
    )
    parser.add_subclass_arguments(RankingMetric, "ranking_metric")
    parser.add_subclass_arguments(UnifiedMetric, "unified_metric")
    parser.add_subclass_arguments(EarlyStopping, "early_stopping")
    parser.add_subclass_arguments(ModelCheckpoint, "model_checkpoint")
    parser.add_subclass_arguments(Trainer, "trainer")
    parser.add_argument(
        "--load_from_checkpoint",
        help="Loads a model checkpoint for fine-tuning",
        default=None,
    )
    parser.add_argument(
        "--strict_load",
        action="store_true",
        help="Strictly enforce that the keys in checkpoint_path match the keys returned by this module's state dict.",
    )
    return parser


def initialize_trainer(configs) -> Trainer:
    checkpoint_callback = ModelCheckpoint(
        **namespace_to_dict(configs.model_checkpoint.init_args)
    )
    early_stop_callback = EarlyStopping(
        **namespace_to_dict(configs.early_stopping.init_args)
    )
    trainer_args = namespace_to_dict(configs.trainer.init_args)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer_args["callbacks"] = [early_stop_callback, checkpoint_callback, lr_monitor]
    # trainer_args['logger'] = wandb_logger
    # wandb_logger.experiment.config.update(trainer_args)
    # print("TRAINER ARGUMENTS: ")
    # print(json.dumps(trainer_args, indent=4, default=lambda x: x.__dict__))
    trainer = Trainer(**trainer_args)
    return trainer


def initialize_model(configs):
    print(
        json.dumps(
            configs.unified_metric.init_args, indent=4, default=lambda x: x.__dict__
        )
    )
    if configs.load_from_checkpoint is not None:
        logger.info(f"Loading weights from {configs.load_from_checkpoint}.")
        model = UnifiedMetric.load_from_checkpoint(
            checkpoint_path=configs.load_from_checkpoint,
            strict=False,
            **namespace_to_dict(configs.unified_metric.init_args),
        )
    else:
        model = UnifiedMetric(**namespace_to_dict(configs.unified_metric.init_args))


    return model


def train_command() -> None:
    parser = read_arguments()
    cfg = parser.parse_args()
    print("Hereeeeeeeeee")
    seed_everything(cfg.seed_everything)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load with explicit map_location
    #model = torch.load(checkpoint_path, map_location=device)
    model = UnifiedMetric.load_from_checkpoint(cfg.ckpt_path, **namespace_to_dict(cfg.unified_metric.init_args), map_location=device, strict=True)
    model.eval()
    examples = []
    with open(cfg.test_file, 'r') as f:
        lines = f.readlines()
        # print("==============================")
        # print(lines)
    for l in lines:
        exp = json.loads(l)
        examples.append(exp)
    #model.load_from_checkpoint(cfg.ckpt_path)
    scores = model.predict(examples, batch_size=100)


    #scores = model.predict(examples, batch_size=100)
    for s, ex in zip(scores.scores, examples):
        ex["predict_score"] = float(s)
            
    write_jsonl(cfg.out_file, examples) 

#python inference.py --cfg configs/models/unified_metric.yaml --ckpt_path /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt --test_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/tests/temp_data.jsonl --out_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/results/out.jsonl
#python inference.py --cfg configs/models/unified_metric.yaml --ckpt_path /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt --test_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/tests/pie_for_codescore.jsonl --out_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/results/out.jsonl
#python inference.py --cfg configs/models/unified_metric.yaml --ckpt_path /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/tests/non_eq_random.jsonl --out_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/results/output_random.jsonl
#python inference.py --cfg configs/models/unified_metric.yaml --ckpt_path /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt --test_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/tests/non_eq_random.jsonl --out_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/results/output_random.jsonl
#python inference.py --cfg configs/models/unified_metric.yaml --ckpt_path /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt --test_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/Datasets/MBPP/test.jsonl --out_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/results/output_mbpp_test.jsonl
#/scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/Datasets/MBPP/test.jsonl

#python inference_orig.py --cfg configs/models/unified_metric_orig.yaml --ckpt_path /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/CodeScore/models/epoch%3D8-step%3D299583-val_pearson%3D0.739.ckpt --test_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/tests/pie_for_codescore.jsonl --out_file /scratch/nwc8gr/Code_Sum_Gen/BaseLInes/Evaluation/CodeScore/results/out.jsonl


if __name__ == "__main__":
    train_command()
