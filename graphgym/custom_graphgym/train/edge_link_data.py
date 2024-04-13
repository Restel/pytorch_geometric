import torch
from torch.utils.data import DataLoader
from typing import Optional
from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.loader import create_dataset, set_dataset_attr
from torch_geometric.transforms import RandomLinkSplit, TemporalLinkSplit
from torch_geometric.loader import LinkNeighborLoader 
from torch_geometric.data import Data
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torch_geometric.io import edge_mask, mask_tensor
from math import floor
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE, AddRemainingSelfLoops



def set_data_attr(data, name, value):
    data[name] = value
    

def set_split_attributes(split_data: Data, split: str) -> Data:
    set_data_attr(split_data, f'{split}_edge_index', split_data['edge_index'])
    set_data_attr(split_data, f'{split}_edge_label_index', split_data['edge_label_index'])
    return split_data

import torch
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.utils import negative_sampling
from typing import Callable
from torch_geometric.data import Dataset

class SingleGraphDataset(Dataset):
    def __init__(self, data: Data, transform: Optional[Callable] = None):
        """
        Initializes the dataset.

        Args:
            data (Data): A single instance of torch_geometric.data.Data.
            transform (Callable, optional): A function that transforms data.
        """
        super(SingleGraphDataset, self).__init__()
        self.data = data
        if transform is not None:
            try: 
                self.data = transform(self.data)
            except Exception:
                print(f"Issue with {transform} augmentation")
        self._validate_dim_in()

    def _validate_dim_in(self):
        if cfg.share.dim_in != self.data.num_features:
            cfg.share.dim_in = self.data.num_features
            print(f'resetting share dim in for GNN model to match dataloader dim:  to {cfg.share.dim_in}')
        

    def len(self):
        """
        Returns the number of graphs in the dataset.
        """
        return 1

    def get(self, idx):
        """
        Gets the graph at a particular index (only one graph in this case).

        Args:
            idx (int): The index of the graph to retrieve.
        
        Returns:
            Data: The graph data.
        """
        return self.data


def collate(data):
    # Assuming all your Data objects should be in a single batch
    # This function can be expanded based on your specific needs,
    # especially if your Data objects require more complex merging
    batch = Data()
    keys = data.keys
    for key in keys:
        #batch[key] = torch.cat([d[key] for d in data], dim=0)
        batch[key] = data[key]
    return batch

class FullBatchLinkDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        # It's important to note that for link prediction tasks where all edges
        # should be in a single batch, the batch_size is typically set to the size
        # of the dataset or 1 if processing the whole dataset at once.
        super(FullBatchLinkDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=collate, **kwargs)


@register_train("CustomGraphGymDataModule")
class CustomGraphGymDataModule(LightningDataModule): 
    def __init__(self, split_type = 'static'):
        super().__init__(has_val=True, has_test=True)
        self.transform = self._get_transforms_from_cfg()
        self.split_type = split_type
        self.dataset = create_dataset()
        self.num_points = len(self.dataset)
        self.number_of_possible_splits = len(self.dataset) - 3 if split_type=='temporal' else 1
        self.splits = [{'train': None, 'val': None, 'test': None} for _ in range(self.number_of_possible_splits)]
        self.current_split = 0
        self._create_dataset_splits(self.split_type)
        self._create_data_loaders()

    def _get_transforms_from_cfg(self):
        transform_list = []
        for t in cfg.dataset.transform:
            print(t)
            if t == 'laplacian_pe':
                transform_list.append(AddLaplacianEigenvectorPE(k=50, attr_name=None))
            elif t == 'random_walk_pe':
                transform_list.append(AddRandomWalkPE(30, attr_name=None))
            else:
                raise ValueError('Unsupported transform')
        if transform_list:
            return Compose(transform_list)
        else:
            return None 

    def _create_dataset_splits(self, split_type):
        add_negative_train_samples = not cfg.dataset.resample_negative
        if split_type == 'static':
            f = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=add_negative_train_samples, neg_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio)
            splits = f(self.dataset[min(0,self.num_points-2)])
            for split_data, split_name in zip(splits, ['train', 'val', 'test']):
                self.splits[0][split_name] = split_data
        elif split_type == 'temporal':
            for i in range(self.number_of_possible_splits):
                f = TemporalLinkSplit(i, add_negative_train_samples=add_negative_train_samples, neg_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio)
                splits = f(self.dataset) 
                for split_data, split_name in zip(splits, ['train', 'val', 'test']):
                    self.splits[i][split_name] = split_data
        else:
            raise ValueError("unsupported type split and dataset configuration")
        if  cfg.share.dim_in == 1:
            cfg.share.dim_in = splits[0].x.shape[1]
            print(f'resetting share dim in for GNN model to: {cfg.share.dim_in}')

    def _get_split_statistics(self):
        import pandas as pd
        statistics = [] # a list of dataframes with split statistics
        index = ['train', 'val', 'test']
        for i in range(len(self.splits)):
            res = {'# MPP edges': [], '# label pos edges': [], '# label neg edges': []}
            for split in index:
                data = self.splits[i][split]
                res['# MPP edges'].append(data.edge_index.size(1))
                num_pos_edges = data.edge_label[data.edge_label == 1].size(0)
                num_neg_edges = data.edge_label[data.edge_label == 0].size(0)
                res['# label pos edges'].append(num_pos_edges)
                res['# label neg edges'].append(num_neg_edges)
            df_res = pd.DataFrame(res, index = index)
            statistics.append(df_res)
        return statistics

                
    
    def _move_to_next_split(self):
        if self.current_split + 1 >= self.number_of_possible_splits:
            print('End of the splits. Resetting to the beginning')
            self.current_split = 0
            self._create_data_loaders()
    
        else:
            self.current_split += 1
            self._create_data_loaders()
            print(f'Moving to the next split: {self.current_split}')

    def _create_data_loader(self, split):
        split_data = self.splits[self.current_split][split]
        split_data.edge_index = split_data.edge_index.contiguous()
        split_data_wrapped = SingleGraphDataset(split_data, transform=self.transform)
        pw = cfg.num_workers > 0
        shuffle = True if split == 'train' else False
        return FullBatchLinkDataLoader(split_data_wrapped, 
                                        batch_size=1, 
                                        num_workers=cfg.num_workers,
                                        pin_memory=True,
                                        persistent_workers=pw,
                                        shuffle=shuffle)

    def _create_data_loaders(self):
        self._train_dataloader = self._create_data_loader('train')
        self._val_dataloader = self._create_data_loader('val')
        self._test_dataloader = self._create_data_loader('test')

    def train_dataloader(self) -> FullBatchLinkDataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> FullBatchLinkDataLoader:
        return self._val_dataloader

    def test_dataloader(self) -> FullBatchLinkDataLoader:
        return self._test_dataloader
    

@register_train("train_pl")
def train(
    model,
    datamodule,
    logger: bool = True,
    trainer_config: Optional[dict] = None,
):
    callbacks = []
    if logger:
        callbacks.append(LoggerCallback())
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
        callbacks.append(ckpt_cbk)

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices=find_usable_cuda_devices(cfg.devices)
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # for future try from pytorch_lightning.accelerators import find_usable_cuda_devices
    # # Find two GPUs on the system that are not already occupied
    # trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(cfg.devices))
    #  pl 2.1.0 https://lightning.ai/docs/pytorch/2.1.0/accelerators/gpu_basic.html

class GlobalEpochCallback(pl.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.global_epoch = 0

    def on_epoch_end(self, trainer, pl_module):
        self.global_epoch += 1
        if self.global_epoch >= self.total_epochs:
            trainer.should_stop = True  # Manually signal to stop training
        
class ResetEpochNumberCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self._epoch = 0
            
    def on_train_start(self, trainer, pl_module):
        trainer.fit_loop.epoch_progress.current.completed = self._epoch 
        print("Training starts with epoch set to",trainer.current_epoch)


class ManageDataSplitsCallback(pl.Callback):
    def __init__(self, total_epochs, splits):
        super().__init__()
        self.total_epochs = total_epochs
        self.splits = splits
        self.epochs_per_split = total_epochs // splits
        self.current_split = 0

    def on_train_epoch_end(self, trainer, pl_module):
        # Determine if it's time to move to the next split
        

        if (trainer.fit_loop.epoch_progress.current.processed + 1) % self.epochs_per_split == 0 and self.current_split < self.splits - 1:
            self.current_split += 1
            print(f"Moving to split {self.current_split}")
            # Logic to update the datamodule to the next split
            trainer.datamodule._move_to_next_split()
            # Optionally reset trainer's internal state if needed
            # Be careful with resetting internal state to avoid unintended side effects

class ResetLearningRateCallback(pl.Callback):
    def __init__(self, initial_lr):
        """
        Args:
            initial_lr (float or list): The initial learning rate(s) of the optimizer(s).
                                        If multiple optimizers are used, pass a list of learning rates.
        """
        super().__init__()
        self.initial_lr = initial_lr

    def on_train_start(self, trainer, pl_module):
        """Event that will be called at the beginning of the training."""
        self.reset_learning_rate(trainer, pl_module)

    def reset_learning_rate(self, trainer, pl_module):
        """Reset the learning rate(s) of the optimizer(s) to the initial value(s)."""
        optimizers = trainer.optimizers

        # Check if initial_lr is a list for handling multiple optimizers
        if isinstance(self.initial_lr, list):
            for opt, lr in zip(optimizers, self.initial_lr):
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
        else:
            # Single learning rate value, assuming single optimizer
            for opt in optimizers:
                for param_group in opt.param_groups:
                    param_group['lr'] = self.initial_lr


@register_train("train_pl_temporal")
def train(
    model,
    datamodule,
    logger: bool = True,
    trainer_config: Optional[dict] = None,
):
    
    # Create the callback
    manage_splits_cb = ManageDataSplitsCallback(
        total_epochs=cfg.optim.max_epoch,
        splits=datamodule.number_of_possible_splits
    )
    epochs_per_split = cfg.optim.max_epoch // datamodule.number_of_possible_splits

    reset_lr_callback = ResetLearningRateCallback(initial_lr=0.01)

    # Include the callback in your callbacks list
    callbacks = [
        manage_splits_cb,
        reset_lr_callback,
    ]
    if logger:
        callbacks.append(LoggerCallback())
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
        callbacks.append(ckpt_cbk)

    trainer_config = trainer_config or {}

    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices=find_usable_cuda_devices(cfg.devices),
        reload_dataloaders_every_n_epochs=epochs_per_split,
    )


    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
        #datamodule._move_to_next_split()
        #trainer.fit_loop.epoch_progress.current.processed = 0
        
        #trainer.fit_loop.done = False
        
        # trainer.fit(model, datamodule=datamodule)
        # trainer.test(model, datamodule=datamodule)
        # datamodule._move_to_next_split()

    # for future try from pytorch_lightning.accelerators import find_usable_cuda_devices
    # # Find two GPUs on the system that are not already occupied
    # trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(cfg.devices))
    #  pl 2.1.0 https://lightning.ai/docs/pytorch/2.1.0/accelerators/gpu_basic.html_temporal