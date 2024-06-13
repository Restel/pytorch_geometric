import logging
import os
import custom_graphgym  # noqa, register custom modules
import torch
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym import register
import optuna
import numpy as np
from pytorch_lightning import seed_everything
import argparse



### Usage:
### python main_spot_checking.py --cfg configs/pyg/ecoli_optuned_static.yaml --study grn-static-convergence-large-set --cutoff 0.9
### python main_spot_checking.py --cfg configs/pyg/ecoli_optuned_temporal.yaml --study grn-temporal-convergence-large-set --cutoff 0.78

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='optuned')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--study', dest='study_name', type=str, required=True,
                        help='The name of the study to create')
    parser.add_argument('--cuda', dest='cuda', type=str, required=False, default='6',
                        help='The cuda device')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

def objective(trial):
    # Define hyperparameters to optimize
    cfg.optim.base_lr = trial.suggest_categorical('base_lr', [0.001, 0.01, 0.05])
    cfg.gnn.layers_mp = trial.suggest_int('layers_mp', 0, 3)
    cfg.optim.max_epoch  = trial.suggest_categorical('max_epoch', [200, 400])
    cfg.gnn.layers_pre_mp = trial.suggest_int('layers_pre_mp', 1, 3)
    cfg.gnn.layers_post_mp = trial.suggest_int('layers_post_mp', 1, 3)
    cfg.gnn.dropout = trial.suggest_categorical('dropout', [0.01, 0.05, 0.1, 0.2, 0.3])
    cfg.gnn.act = trial.suggest_categorical('act', ['relu', 'prelu'])
    cfg.gnn.layer_type = trial.suggest_categorical('layer_type', ['gatconv', 'gcnconv'])
    cfg.gnn.stage_type =  trial.suggest_categorical('stage_type', ['skipsum', 'skipconcat'])
    cfg.gnn.agg =  trial.suggest_categorical('agg', ['add', 'mean', 'max'])
    cfg.gnn.keep_edge = trial.suggest_categorical('keep_edge', [0, 0.1, 0.3, 0.5])
    cfg.gnn.dim_inner =  trial.suggest_categorical('dim_inner', [8, 16, 32, 64])
    cfg.optim.weight_decay = trial.suggest_categorical('weight_decay', [5e-5, 1e-4, 5e-4])
    cfg.optim.scheduler = trial.suggest_categorical('scheduler', ['cos', 'none'])
    cfg.seed = trial.suggest_int('seed', 0, 100)
    
    # Create PyTorch Lightning model with hyperparameters specified in cfg 
    seed_everything(cfg.seed, workers=True)
    if cfg.dataset.name in ['yeast-ppi']:
        print('Loading BIOGRID') 
        datamodule = register.train_dict["BioGridGraphGymDataModule"](split_type = cfg.dataset.split_type)    
    else:
        print('Loading Abasy') 
        datamodule = register.train_dict["CustomGraphGymDataModule"](split_type = cfg.dataset.split_type)
    model = create_model()
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    # Train model using PyTorch Lightning Trainer
    val_metric = register.train_dict["train_optuna"](model, datamodule, logger=True)
    return val_metric

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    dump_cfg(cfg)
    set_printing()
    auto_select_device() # if not set in the yaml config, set to cuda accelerator if available and single device
    # Set machine learning pipeline
    cfg.share.dim_out = 1
    cfg.share.num_splits = 3

    # Define Optuna study
    study = optuna.create_study(direction='maximize',
                                study_name = args.study_name,
                                storage='sqlite:///yeast.db',
                                load_if_exists=True)

    study.optimize(objective, n_trials=500)