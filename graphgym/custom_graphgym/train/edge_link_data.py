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
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader 
from torch_geometric.data import Data

def set_data_attr(data, name, value):
    data[name] = value
    

def set_split_attributes(split_data: Data, split: str) -> Data:
    set_data_attr(split_data, f'{split}_edge_index', split_data['edge_index'])
    set_data_attr(split_data, f'{split}_edge_label_index', split_data['edge_label_index'])
    return split_data

@register_train("CustomGraphGymDataModule")
class CustomGraphGymDataModule(LightningDataModule): 
    def __init__(self):
        super().__init__(has_val=True, has_test=True)
        self.dataset = create_dataset()[0]
        self.splits = {'train': None, 'test': None, 'val': None}
        self._create_dataset_splits()
        self._create_data_loaders()

    def _create_dataset_splits(self):
       #dataset = load_dataset() # TODO modify load_dataset for GRN
#         f = T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
#                       add_negative_train_samples=False, neg_sampling_ratio=0)
# train_data, val_data, test_data = f(data_pure)
        add_negative_train_samples = not cfg.dataset.resample_negative
        f = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                       add_negative_train_samples=False, neg_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio) # TODO add negative samples during the training???
        splits = f(self.dataset) # TODO add cfg option and register it?
        # for split_data, split_name in zip(splits, ['train', 'val', 'test']):
        #     self.splits[split_name] = set_split_attributes(split_data, split_name)
        for split_data, split_name in zip(splits, ['train', 'val', 'test']):
            self.splits[split_name] = split_data

        # id_all = torch.cat([id, id_neg], dim=-1)
        # label = create_link_label(id, id_neg)
        # set_dataset_attr(dataset, 'train_edge_index', id_all,
        #                     id_all.shape[1])
        
        
        # neg_sampling = True,
        # neg_sampling_ratio=cfg.dataset.edge_negative_sampling_ratio,
    def _create_data_loaders(self):
        pw = cfg.num_workers > 0
        self._train_dataloader = LinkNeighborLoader(
            data=self.splits['train'],
            num_neighbors=[-1],
            batch_size=self.dataset.num_edges,
            edge_label_index= self.splits['train'].edge_label_index,
            edge_label= self.splits['train'].edge_label,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=pw
        )

        self._val_dataloader = LinkNeighborLoader(
            data=self.splits['val'],
            num_neighbors=[-1],
            batch_size=self.dataset.num_edges,
            edge_label_index= self.splits['val'].edge_label_index,
            edge_label= self.splits['val'].edge_label,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=pw
        )

        self._test_dataloader = LinkNeighborLoader(
            data=self.splits['test'],
            num_neighbors=[-1],
            batch_size=self.dataset.num_edges,
            edge_label_index= self.splits['test'].edge_label_index,
            edge_label= self.splits['test'].edge_label,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=pw
        )

    def train_dataloader(self) -> LinkNeighborLoader:
        return self._train_dataloader

    def val_dataloader(self) -> LinkNeighborLoader:
        return self._val_dataloader

    def test_dataloader(self) -> LinkNeighborLoader:
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
        devices="auto" if not torch.cuda.is_available() else cfg.devices,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # for future try from pytorch_lightning.accelerators import find_usable_cuda_devices
    # # Find two GPUs on the system that are not already occupied
    # trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(cfg.devices))
    #  pl 2.1.0 https://lightning.ai/docs/pytorch/2.1.0/accelerators/gpu_basic.html