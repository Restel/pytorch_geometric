from torch_geometric.data import Data
import os
import json
from torch_geometric.data.makedirs import makedirs
import numpy as np
import torch

from typing import List, Optional, Callable, Tuple


grn_files = {'Ecoli': ['511145_v2003_sRDB01',  
                       '511145_v2005_sRDB04', 
                       '511145_v2006_sRDB06',
                       '511145_v2011_sRDB11', 
                       '511145_v2013_sRDB13', 
                       '511145_v2014_sRDB16',
                       '511145_v2015_sRDB16', 
                       '511145_v2017_sRDB16', 
                       '511145_v2018_sRDB19', 
                       '511145_v2020_s13-RDB19',
                       '511145_v2022_s13-RDB22']}


def convert_labels(grn:List[tuple[str]]) -> List[tuple[str]]:
    """Convert raw string edge labels into integers

    Parameters
    ----------
    grn : List[tuple[str]]
        a list of edge triplets in form (source, target, sign) for one graph

    Returns
    -------
    List[tuple[str]]
        list of edge triples with edge signs converted to integer and split doble edges

    Raises
    ------
    ValueError
        Unsupported edge sign
    """
    questioned = 0
    new_label_list = []
    for s,t,sign in grn:
        if sign == '-':
            new_label_list.append((s,t, -1))
        elif sign == '?':
            questioned += 1
        elif sign == '+':
            new_label_list.append((s,t, 1))
        elif sign == '+-' or sign == '-+':
            new_label_list.append((s,t, -1))
            new_label_list.append((s,t, 1))
        else:
            raise ValueError(f"Unsupported type {sign}")
    print(f'Filtered {questioned} edges of unknown type')
    return new_label_list


def read_grn_data(dir:str, name:str) -> Tuple[List[Data], List[str]]:

    # Input validation
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory '{dir}' does not exist.")
    
    if name not in grn_files:
        raise ValueError(f"Name '{name}' not found in grn_files.")
    

    num_graphs = len(grn_files[name])
    datalist = [] # list with graphs in Data format
    node_list = []
    edge_list = []
    unique_nodes = set()
    
    years = []

    for file in grn_files[name]:
         filename = os.path.join(dir, f'{file}.json')
         print(filename)
         year = file.split("_")[1]
         years.append(year)
         with open(filename, 'r') as json_file:
            data = json.load(json_file)
            node_ids = [node['data']['id'] for node in data['elements']['nodes']]
            node_list.append(node_ids)
            unique_nodes.update(set(node_ids))

            edges = []
            edges = [(edge['data']['source'], edge['data']['target'], edge['data']['Effect']) for edge in data['elements']['edges']]
            converted_edges = convert_labels(edges)
            edge_list.append(converted_edges)

    # map nodes in each grn dataset and edge to its global unique id 
    node_id_list = list(unique_nodes)
    node_id_mapping = {id_str: idx for idx, id_str in enumerate(node_id_list)}

    edge_list_mapped = [
        [(node_id_mapping[x[0]], node_id_mapping[x[1]]) for x in grn] for grn in edge_list
    ]
    edge_labels = [[x[2] for x in grn] for grn in edge_list]
    node_list_mapped = [[node_id_mapping[x] for x in grn] for grn in node_list]

    # one-hot encoding as base features
    X = torch.eye(len(node_id_list), dtype=torch.float)

    for graph in range(num_graphs):
        edge_idx = torch.tensor(np.array(edge_list_mapped[graph]).transpose(), dtype=torch.long)
        edge_attr = torch.tensor(edge_labels[graph], dtype=torch.long)
        data = Data(x=X, edge_attr=edge_attr, y=edge_attr, edge_index=edge_idx)
        datalist.append(data)

    return datalist, years

def edge_mask(edge_index1, edge_index2):
    """Filter edges from edge_index1 that are in edge_index2 """
    # Initialize mask with True values
    mask = torch.ones(edge_index1.size(1), dtype=torch.bool)
    
    # Convert edge_index2 to a set for efficient containment checks
    set2 = {(edge_index2[0, i].item(), edge_index2[1, i].item()) for i in range(edge_index2.size(1))}
    
    # Iterate over edges in edge_index1 and check if they exist in edge_index2
    for i in range(edge_index1.size(1)):
        edge = (edge_index1[0, i].item(), edge_index1[1, i].item())
        if edge in set2:
            mask[i] = False
    
    return mask

def mask_tensor(tensor, mask):
    if len(tensor.size()) == 2:
        return tensor[:, mask].squeeze()
    elif len(tensor.size()) == 1:
        return tensor[mask]
    else:
        raise AttributeError("Dimensions dont match")