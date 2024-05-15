import requests
import torch
from torch_geometric.data import InMemoryDataset, Data
import requests
import os
import json
from torch_geometric.data.makedirs import makedirs
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.utils import to_networkx

from typing import List, Optional, Callable, Tuple


grn_files = {'Ecoli': ['511145_v2005_sRDB04', 
                       '511145_v2011_sRDB11', 
                       '511145_v2015_sRDB16', 
                       '511145_v2017_sRDB16', 
                       '511145_v2018_sRDB19', 
                       '511145_v2020_s13-RDB19',
                       '511145_v2022_s13-RDB22']}

def edge_mask(edge_index1: torch.tensor, edge_index2:torch.tensor)-> torch.tensor:
    """Returns the boolean mask to filter out edges in edge_index1 that 
    are present in edge_index2

    Parameters
    ----------
    edge_index1 : torch.tensor
        _description_
    edge_index2 : torch.tensor
        _description_

    Returns
    -------
    torch.tensor
        Mask for edge_index1 = True if edge is not edge_index2
    """
    # Initialize mask with False values
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

def split_stats(train_data, val_data, test_data):
    train_m = train_data.edge_index.size(1)
    train_s = train_data.edge_label_index.size(1)

    val_m = val_data.edge_index.size(1)
    val_s = val_data.edge_label_index.size(1)

    test_m = test_data.edge_index.size(1)
    test_s = test_data.edge_label_index.size(1)
    test = test_m + test_s
    print(f"Training set: \t Number of message edges {train_m} ({train_m/test:.3f} % from the final graph)")
    print(f"Number of supervision edges {train_s} ({train_s/test:.3f} % from the final graph):")
    print(f"Validation set: \t Number of message edges {val_m} ({val_m/test:.3f} % from the final graph)")
    print(f"Number of supervision edges {val_s} ({val_s/test:.3f} % from the final graph)")
    print(f"Test set: \t Number of message edges {test_m} ({test_m/test:.3f} % from the final graph)")
    print(f"Number of supervision edges {test_s} ({test_s/test:.3f} % from the final graph)")

def toUnsigned(grn:Data) -> Data:
    """A pre-transform function to return an unsigned graph

    Parameters
    ----------
    grn : Data
        A signed GRN graph in torch.utils.Data form with edge_attr containing the sign of the edge 

    Returns
    -------
    Data
        An unsigned GRN graph with modified target
    """
    y = torch.ones_like(grn.edge_attr)
    grn.y = y
    return grn

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
        #n_nodes = len(node_list_mapped[graph])
        #x = X[node_list_mapped[graph]][:, node_list_mapped[graph]] # extract the eye matrix for nodes within the graph and pad with remaining zeros in features
        #pad = torch.zeros((n_nodes, len(unique_nodes)-n_nodes), dtype = torch.float)
        # print(x.size())
        # print(pad.size())
        # x = torch.cat((x,pad), dim = -1)
        # print(x.size())
        
    # print(datalist[0].size())
    # print(datalist[1].size())
    # print(datalist[2].size())
    
    return datalist, years


class GRNDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        self.name = name
        super(GRNDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # self.load(self.processed_paths[0]) # for pyg >= 2.4
        self.data_list = torch.load(self.processed_paths[0])
        self.versions = torch.load(self.processed_paths[1])
    
    @property
    def raw_dir(self) -> str:
        name = os.path.join(self.root, self.name, 'raw')
        return name

    @property
    def processed_dir(self) -> str:
        name = os.path.join(self.root, self.name, 'processed')
        return name

    @property
    def raw_file_names(self):
        # Return a list of raw file names (if any)
        return grn_files[self.name]

    @property
    def processed_file_names(self):
        # Return a list of processed file names (if any)
        return ['data.pt', 'versions.pt']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        current_graph = self.data_list[idx]
        if self.transform is not None:
            current_graph = self.transform(current_graph)
        return current_graph

    def compute_statistics(self)-> pd.DataFrame:
        # Todo: fix the num_nodes, add high confidence vs low confidence edges, 
        # % of removed edges (retracted), % of new edges, % of confirmed edges that transitioned from 
        # low confidence into high confidence, average diameter, and other network charachteristics
        stats = {
            'year': [],
            'num_nodes': [],
            'num_edges': [],
            'num_self_loops': [],
            'num_connected_components': [],
            'avg_degree': [],
            'validated': []
        }

        for graph, ver in zip(self.data_list, self.versions):
            nx_graph = to_networkx(graph)
            G_undir = nx.Graph()
            G_undir.add_edges_from(graph.edge_index.t().tolist())
            stats['year'].append(ver[1:])
            stats['num_nodes'].append(graph.num_nodes)
            stats['num_edges'].append(graph.num_edges)
            stats['num_self_loops'].append(sum(1 for u, v in nx_graph.edges() if u == v))
            stats['num_connected_components'].append(nx.number_connected_components(G_undir))
            stats['avg_degree'].append(graph.num_edges / graph.num_nodes)  # Assuming directed graph
            stats['validated'].append(graph.validate(raise_on_error=True)) 
            
        return pd.DataFrame(stats)

    def download(self):
        # Download the raw data and save it to the raw directory
        headers = {
            'User-Agent': f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        } # TODO user agent getter e.g https://www.whatismybrowser.com/detect/what-is-my-user-agent/
        api_stem = 'https://abasy.ccg.unam.mx/rest/regnets/'
        for file in grn_files[self.name]:
            api_url = f'{api_stem}{file}?field=Regnet&format=json'
            response = requests.get(api_url, headers=headers)

            if response.status_code == 200:
                # Access the response content as json
                data = response.json()

                json_file_path = os.path.join(self.raw_dir, f'{file}.json')
                
                makedirs(self.raw_dir)
                with open(json_file_path, 'w') as json_file:
                   json.dump(data, json_file)

            else:
                print(f"Failed to retrieve data. Status Code: {response.status_code}")


    def process(self):
        # Process the raw data and save it to the processed directory
        data_list, years  = read_grn_data(self.raw_dir, self.name)
        self.versions = years
        print(data_list)
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
            # 1. TODO write pre-filter func to remove dual edges 
            

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list] 
            # 2. TODO write pre-tansform or transform (?) function to make edges unsigned
            # 3. TODO write pre-tansform or transform (?) function to filter out weak edges
        
        #self.data, self.slices = self.collate(data_list)
        self.data_list = data_list
        print(self.data_list)
        # Store the processed data
        torch.save(self.data_list, self.processed_paths[0])
        torch.save(self.versions, self.processed_paths[1])


#import requests
# # Complete API URL
# #api_url = 'https://abasy.ccg.unam.mx/rest/complexes/511145_v2005_sRDB04?format=json'
# api_stem = 'https://abasy.ccg.unam.mx/rest/regnets/'
# api_url = 'https://abasy.ccg.unam.mx/rest/regnets/511145_v2005_sRDB04?field=Regnet&format=json'
# api_url2 = '511145_v2011_sRDB11?field=Regnet&format=json'
# api_url3 = '511145_v2022_sRDB22_eStrong'
# api_url = f'{api_stem}{api_url3}?field=Regnet&format=json'
# headers = {
#     'User-Agent': f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     # Add any other headers as needed
# }

# response = requests.get(api_url, headers=headers)

# if response.status_code == 200:
#     # Access the response content as text
#     data = response.json()

#     # Process the data as needed
#     print(data)
# else:
#     print(f"Failed to retrieve data. Status Code: {response.status_code}")