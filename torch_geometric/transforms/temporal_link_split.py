import copy
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData, Dataset
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import EdgeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType
from torch_geometric.utils import negative_sampling
from torch_geometric.io import edge_mask, mask_tensor

@functional_transform('temporal_link_split')
class TemporalLinkSplit(BaseTransform):
    def __init__(
        self,
        i,
        key: str = 'edge_label',
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
    ) -> None:
        self.i = i
        self.key = key
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio

    def forward(
        self,
        data: Dataset,
    ) -> Tuple[
            Data,
            Data,
            Data,
    ]:
        assert len(data) > self.i + 3
        

        train_data = copy.copy(data[self.i])
        val_data = copy.copy(data[self.i+1])
        test_data = copy.copy(data[self.i+2])
        test_label_data = copy.copy(data[self.i+3])

        
        assert isinstance(train_data, Data)
        assert isinstance(val_data, Data)
        assert isinstance(test_data, Data)
        assert isinstance(test_label_data, Data)

        #train_edges = train_data.edge_index
        train_edges = train_data.edge_index  
        val_edges = val_data.edge_index
        test_edges = test_data.edge_index
        test_label_edges = test_label_data.edge_index
        
        train_label_edges_index = edge_mask(val_edges, train_edges)
        val_label_edges_index = edge_mask(test_edges, val_edges)
        test_label_edges_index = edge_mask(test_label_edges, test_edges)

        
        num_train = train_label_edges_index.sum().item() # number of supervision edges
        num_val = val_label_edges_index.sum().item()
        num_test = test_label_edges_index.sum().item()

        num_neg_train = int(num_train * self.neg_sampling_ratio)
        num_neg_val = int(num_val * self.neg_sampling_ratio)
        num_neg_test = int(num_test * self.neg_sampling_ratio)

        num_neg = num_neg_train + num_neg_val + num_neg_test

        neg_edge_index = negative_sampling(train_edges, 
                                               num_neg_samples=num_neg,
                                               method='sparse')

        # Adjust ratio if not enough negative edges exist
        if neg_edge_index.size(1) < num_neg:
            num_neg_found = neg_edge_index.size(1)
            ratio = num_neg_found / num_neg
            warnings.warn(
                f"There are not enough negative edges to satisfy "
                "the provided sampling ratio. The ratio will be "
                f"adjusted to {ratio:.2f}.")
            num_neg_train = int((num_neg_train / num_neg) * num_neg_found)
            num_neg_val = int((num_neg_val / num_neg) * num_neg_found)
            num_neg_test = num_neg_found - num_neg_train - num_neg_val

        
        self._create_label(
            val_data,
            train_label_edges_index,
            neg_edge_index[:, num_neg_val + num_neg_test:],
            out=train_data,
        )
        self._create_label(
            test_data,
            val_label_edges_index,
            neg_edge_index[:, :num_neg_val],
            out=val_data,
        )
        self._create_label(
            test_label_data,
            test_label_edges_index,
            neg_edge_index[:, num_neg_val:num_neg_val + num_neg_test],
            out=test_data,
        )

        return train_data, val_data, test_data


    def _create_label(
        self,
        store: EdgeStorage,
        index: Tensor,
        neg_edge_index: Tensor,
        out: EdgeStorage,
    ) -> EdgeStorage:
        index = index.nonzero(as_tuple=True)[0].contiguous()
        edge_index = store.edge_index[:, index].contiguous()
        print(f'Positive edges {edge_index.size(1)}')

        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index].contiguous()
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            if neg_edge_index.numel() > 0:
                assert edge_label.dtype == torch.long
                assert edge_label.size(0) == edge_index.size(1)
                edge_label.add_(1)
            if hasattr(out, self.key):
                delattr(out, self.key)
        else:
            edge_label = torch.ones(edge_index.size(1), device=index.device)

        if neg_edge_index.numel() > 0:
            neg_edge_label = edge_label.new_zeros((neg_edge_index.size(1), ) +
                                                  edge_label.size()[1:])
            print(f'Negative edges {neg_edge_label.size(0)}')

        if self.split_labels:
            out[f'pos_{self.key}'] = edge_label
            out[f'pos_{self.key}_index'] = edge_index
            if neg_edge_index.numel() > 0:
                out[f'neg_{self.key}'] = neg_edge_label
                out[f'neg_{self.key}_index'] = neg_edge_index

        else:
            if neg_edge_index.numel() > 0:
                edge_label = torch.cat([edge_label, neg_edge_label], dim=0).contiguous()
                edge_index = torch.cat([edge_index, neg_edge_index], dim=-1).contiguous()
            out[self.key] = edge_label
            out[f'{self.key}_index'] = edge_index

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_val={self.num_val}, '
                f'num_test={self.num_test})')
