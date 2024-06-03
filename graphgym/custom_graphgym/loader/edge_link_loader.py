from torch_geometric.datasets import QM7b, Planetoid, GRNDataset, BioGridDataset
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.config import cfg
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data import Data

@functional_transform('remove_self_loops')
class RemoveSelfLoops(BaseTransform):
    r"""Removes self loops
    """
    
    def forward(
        self,
        data: Data,
    ) -> Data:
        
        mask = data.edge_index[0] != data.edge_index[1]
        data.edge_index = data.edge_index[:, mask]
        # Check if edge_attr exists before applying the mask
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[mask]
        return data

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
        elif name in ['Ecoli']:
            f = T.Compose([T.RemoveDuplicatedEdges(), RemoveSelfLoops()])
            dataset = GRNDataset(dataset_dir, name, pre_transform=f)
        elif name in ['yeast-ppi', 'human-ppi']:
            f = T.Compose([T.RemoveDuplicatedEdges(), RemoveSelfLoops()])
            dataset = BioGridDataset(dataset_dir, name, transform=f)
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