import logging
import os

import custom_graphgym  # noqa, register custom modules
import torch

import logging
import os
import custom_graphgym  # noqa, register custom modules
import torch
from pytorch_lightning import seed_everything
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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,7"
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    dump_cfg(cfg)
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir, i)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed, workers=True)
        auto_select_device() # if not set in the yaml config, set to cuda accelerator if available and single device
        # Set machine learning pipeline
        datamodule = register.train_dict["CustomGraphGymDataModule"](split_type = cfg.dataset.split_type)
        cfg.share.dim_out = 1 # TODO fix this bug, that happend in set_dataset_info because dataset._data.y might have node labels (not edge labels)
        cfg.share.num_splits = 3 # TODO fix this bug
        model = create_model()
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Call the custom training function
        register.train_dict["train_pl"](model, datamodule, logger=True)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
