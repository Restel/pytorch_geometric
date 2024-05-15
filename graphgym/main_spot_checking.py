import logging
import os
import custom_graphgym  # noqa, register custom modules
import torch
from pytorch_lightning import seed_everything
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
import pandas as pd
from torch_geometric.graphgym import register
import argparse
import optuna
from pytorch_lightning import seed_everything

### Usage:
### python main_spot_checking.py --cfg configs/pyg/ecoli_optuned_static.yaml --study grn-static-convergence-large-set --cutoff 0.9
### python main_spot_checking.py --cfg configs/pyg/ecoli_optuned_temporal.yaml --study grn-temporal-convergence-large-set --cutoff 0.78

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='optuned')

    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('--study', dest='study_name', type=str, required=True,
                        help='The name of the study to load')
    parser.add_argument('--cutoff', dest='cutoff', type=float, required=True,
                        help='The cutoff for selecting the top performing models')
    parser.add_argument('--cuda', dest='cuda', type=str, required=False, default='6',
                        help='The cuda device')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    # Load the study
    print(args.cutoff)
    study=optuna.load_study(study_name=args.study_name,
                    storage='sqlite:///grn.db')
    trials = [t for t in study.trials if t.values and t.values[0] > args.cutoff]
    #sort trials by the performance value
    trials = sorted(trials, key=lambda t: t.values[0], reverse=True)
    trials = [t.params for t in trials]
    df_trials = pd.DataFrame(trials)
    df_trials = df_trials.drop_duplicates()
    print(df_trials)
    grouped_df = df_trials.groupby(df_trials.columns.difference(['seed']).tolist(), as_index=False)['seed'].agg(list)
    unique_exp = grouped_df.to_dict('records') # a list of dictionary with experiments and seeds
    stem = {'base_lr': cfg.optim,
        'layers_mp': cfg.gnn,
        'max_epoch': cfg.optim,
        'layers_pre_mp': cfg.gnn,
        'layers_post_mp': cfg.gnn,
        'dropout': cfg.gnn,
        'act': cfg.gnn,
        'layer_type': cfg.gnn,
        'stage_type': cfg.gnn,
        'agg': cfg.gnn,
        'keep_edge': cfg.gnn,
        'dim_inner': cfg.gnn,
        'weight_decay': cfg.optim,
        'scheduler': cfg.optim,
        'seed': cfg} 
    
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

    metrics = []
    for exp_id, exp_param_set in enumerate(unique_exp[:10]): # Top-10 models
        for seed in exp_param_set['seed']:
            set_run_dir(cfg.out_dir, f'{exp_id}_{seed}')
            for key, val in exp_param_set.items(): # update the parameters
                if key == 'seed':
                    stem[key][key]= seed    
                else:
                    stem[key][key]= val
            print(f'setting the seed:{cfg.seed}')
            seed_everything(cfg.seed, workers=True)
            datamodule = register.train_dict["CustomGraphGymDataModule"](split_type = cfg.dataset.split_type)
            model = create_model()
            cfg.params = params_count(model)
            logging.info('Num parameters: %s', cfg.params)
            # Train model using PyTorch Lightning Trainer
            val_metric = register.train_dict["train_optuna"](model, datamodule, logger=True)
            metrics.append((exp_param_set, val_metric))