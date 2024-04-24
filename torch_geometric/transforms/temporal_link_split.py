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
import random
from torch_geometric.transforms import RandomLinkSplit

@functional_transform('temporal_link_split')
class TemporalLinkSplit(BaseTransform):
    def __init__(
        self,
        indices,
        key: str = 'edge_label',
        split_labels: bool = False,
        add_negative_train_samples: bool = True,
        neg_sampling_ratio: float = 1.0,
    ) -> None:
        self.indices = indices
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
        assert len(self.indices) == 4
        
        
        data_0 = copy.copy(data[self.indices[0]])
        data_1 = copy.copy(data[self.indices[1]])
        data_2 = copy.copy(data[self.indices[2]])
        data_3 = copy.copy(data[self.indices[3]])
        
        
        train_data = Data()
        val_data = Data()
        test_data = Data()
        final_test_data = Data()

        # TODO add negative edges to train
        if self.add_negative_train_samples:
            raise NotImplementedError('Negative sampling for train not supported yet')

        # train construction
        f = RandomLinkSplit(num_val=0, num_test=0, 
                            is_undirected=False,
                            add_negative_train_samples=self.add_negative_train_samples, 
                            disjoint_train_ratio=0.999,
                            neg_sampling_ratio=self.neg_sampling_ratio)
        
        train_data, _, _ = f(data_0)
        #train_data.edge_index = data_0.edge_index
        #train_data.x = data_0.x
        #train_data.edge_attr = data_0.edge_attr
        #train_data.edge_label_index = train_data.edge_index # TODO split train into MPP and supervision randomly if cfg.disjoint_ratio
        #train_data.edge_label = torch.ones(train_data.edge_label_index.size(1)) # all train MPP edges are incuded in supervision

        
        # val construction
        val_data.edge_index = data_0.edge_index # val MPP = train MPP + supervision edges
        val_data.x = train_data.x # val MPP = train MPP edges
        val_data.edge_attr = data_0.edge_attr
        val_label_edges_mask = edge_mask(data_1.edge_index, val_data.edge_index) # val supervision edges
        num_val = val_label_edges_mask.sum().item()
        num_neg_val = int(num_val * self.neg_sampling_ratio)
        neg_edge_index_val = negative_sampling(val_data.edge_index, 
                                               num_neg_samples=num_neg_val,
                                               method='sparse')

        # proceed with label construction using data_1.edge_index as input, 
        # val_label_edges_mask, neg_edge_index_val as index val_data as output
        self._create_label(
            data_1,
            val_label_edges_mask,
            neg_edge_index_val,
            out=val_data,
        ) 

        # test construction
        test_data.edge_index = data_1.edge_index 
        test_data.x = data_1.x 
        test_label_edges_mask = edge_mask(data_2.edge_index, test_data.edge_index) # val supervision edges
        num_test = test_label_edges_mask.sum().item()
        num_neg_test = int(num_test * self.neg_sampling_ratio)
        neg_edge_index_test = negative_sampling(test_data.edge_index, 
                                               num_neg_samples=num_neg_test,
                                               method='sparse')

        # TODO proceed with label construction using data_2.edge_index as input, 
        # test_label_edges_mask, neg_edge_index_test as index test_data as output
        self._create_label(
            data_2,
            test_label_edges_mask,
            neg_edge_index_test,
            out=test_data,
        )

        # final_test construction
        final_test_data.edge_index = test_data.edge_index # val MPP = train MPP edges
        final_test_data.x = test_data.x # val MPP = train MPP edges
        final_test_label_edges_mask = edge_mask(data_3.edge_index, final_test_data.edge_index) 
        num_final_test = final_test_label_edges_mask.sum().item()
        num_neg_final_test = int(num_final_test * self.neg_sampling_ratio)
        neg_edge_index_final_test = negative_sampling(final_test_data.edge_index, 
                                               num_neg_samples=num_neg_final_test,
                                               method='sparse')
        self._create_label(
            data_3,
            final_test_label_edges_mask,
            neg_edge_index_final_test,
            out=final_test_data,
        )
 
        # TODO Adjust ratio if not enough negative edges exist




        
        return train_data, val_data, test_data, final_test_data


    def _split(self, store:EdgeStorage, index: Tensor,) -> EdgeStorage:
        edge_attrs = {key for key in store.keys() if store.is_edge_attr(key)}
        for key, value in store.items():
            if key == 'edge_index':
                continue

            if key in edge_attrs:
                value = value[index]
                store[key] = value
        edge_index = store.edge_index[:, index]
        store.edge_index = edge_index

        return store

    def _create_label(self,
                      store: EdgeStorage,
                      index: Tensor,
                      neg_edge_index: Tensor,
                      out: EdgeStorage,) -> EdgeStorage:
        index = index.nonzero(as_tuple=True)[0].contiguous()
        edge_index = store.edge_index[:, index].contiguous()
        print(f'Positive edges {edge_index.size(1)}')

        if hasattr(store, self.key):
            edge_label = store[self.key]
            edge_label = edge_label[index].contiguous()
            # Increment labels by one. Note that there is no need to increment
            # in case no negative edges are added.
            # if neg_edge_index.numel() > 0:
            #     assert edge_label.dtype == torch.long
            #     assert edge_label.size(0) == edge_index.size(1)
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
    
    

        

@functional_transform('temporal_iterative_link_split')
class TemporalIterLinkSplit(BaseTransform):
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
        
        train_label_edges_mask = edge_mask(val_edges, train_edges) # filter val edges that are in train
        val_label_edges_mask = edge_mask(test_edges, val_edges)
        test_label_edges_mask = edge_mask(test_label_edges, test_edges)

        #train_label_edges_index = torch.tensor([True if x and random.random()<0.25 else False for x in train_label_edges_index]) # downsampling
        
        
        num_train = train_label_edges_mask.sum().item() # number of supervision edges
        num_val = val_label_edges_mask.sum().item()
        num_test = test_label_edges_mask.sum().item()


        num_neg_train = 0
        if self.add_negative_train_samples:
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
            train_label_edges_mask,
            neg_edge_index[:, num_neg_val + num_neg_test:],
            out=train_data,
        )

        #val_label_edges_index_easy = torch.tensor([not x for x in edge_mask(test_edges, train_data.edge_label_index)]) # for debugging add the edges from train label that are in the test edge_ind 2897
        

        self._create_label(
            test_data,
            val_label_edges_mask,
            neg_edge_index[:, :num_neg_val],
            out=val_data,
        )
        self._create_label(
            test_label_data,
            test_label_edges_mask,
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
            # if neg_edge_index.numel() > 0:
            #     assert edge_label.dtype == torch.long
            #     assert edge_label.size(0) == edge_index.size(1)
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
