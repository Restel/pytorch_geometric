from torch_geometric.datasets import QM7b, Planetoid, GRNDataset
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.config import cfg
import torch_geometric.transforms as T

@register_loader('custom')
def load_dataset_example(format, name, dataset_dir):
    print("DOWNLOADING CUSTOM DATASET LOADER")
    dataset_dir = f'{dataset_dir}/{name}'
    if format == 'PyG':
        if name == 'QM7b':
            dataset_raw = QM7b(dataset_dir)
            return dataset_raw
        elif name in ['Cora', 'CiteSeer', 'PubMed']:
            #dataset = Planetoid(dataset_dir, name, transform = T.NormalizeFeatures()) # TODO make a config parameter 
            dataset = Planetoid(dataset_dir, name) 
        if name in ['Ecoli']:
            dataset = GRNDataset(dataset_dir, name)
        else:
            raise ValueError(f"'{name}' is not supported")

    if cfg.dataset.task == 'link_pred':
        for data in dataset.data:
            try: 
                delattr(data, 'y')
                delattr(data, 'train_mask')
                delattr(data, 'val_mask')
                delattr(data, 'test_mask') # for node-classification datasets
            except: 
                Exception
    return dataset