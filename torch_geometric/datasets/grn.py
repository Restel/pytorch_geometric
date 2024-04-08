from torch_geometric.data import InMemoryDataset
import requests
from torch_geometric.io import read_grn_data, convert_labels, grn_files
import os
import json
from torch_geometric.data.makedirs import makedirs
from pandas import DataFrame
import networkx as nx
from torch_geometric.utils import to_networkx
import torch

from typing import List, Optional, Callable, Tuple




class GRNDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        self.name = name
        super(GRNDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # self.load(self.processed_paths[0]) # for pyg >= 2.4
        self.data = torch.load(self.processed_paths[0])
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
        return len(self.data)

    def __getitem__(self, idx):
        current_graph = self.data[idx]
        if self.transform is not None:
            current_graph = self.transform(current_graph)
        return current_graph

    def compute_statistics(self)-> DataFrame:
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

        for graph, ver in zip(self.data, self.versions):
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
            
        return DataFrame(stats)

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
        data, years  = read_grn_data(self.raw_dir, self.name)
        self.versions = years
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]
            # 1. TODO write pre-filter func to remove dual edges 
            

        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data] 
            # 2. TODO write pre-tansform or transform (?) function to make edges unsigned
            # 3. TODO write pre-tansform or transform (?) function to filter out weak edges
        
        #self.data, self.slices = self.collate(data)
        self.data = data
        # Store the processed data
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.versions, self.processed_paths[1])