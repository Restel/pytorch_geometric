from torch_geometric.data import InMemoryDataset, extract_zip
import requests
from torch_geometric.io import read_biogrid_data, convert_labels, grid_files
import os
import json
from torch_geometric.data.makedirs import makedirs
from pandas import DataFrame
import networkx as nx
from torch_geometric.utils import to_networkx
import torch

from typing import List, Optional, Callable, Tuple




class BioGridDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        self.name = name
        super(BioGridDataset, self).__init__(root, transform, pre_transform, pre_filter)
        # self.load(self.processed_paths[0]) # for pyg >= 2.4
        self.data = torch.load(self.processed_paths[0])
        self.versions = torch.load(self.processed_paths[1])
    
    @property
    def raw_dir(self) -> str:
        name = os.path.join(self.root, str(self.name), 'raw')
        return name

    @property
    def processed_dir(self) -> str:
        name = os.path.join(self.root, str(self.name), 'processed')
        return name

    @property
    def raw_file_names(self):
        # Return a list of raw file names (if any)
        return grid_files

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
            stats['year'].append(ver)
            stats['num_nodes'].append(graph.num_nodes)
            stats['num_edges'].append(graph.num_edges)
            stats['num_self_loops'].append(sum(1 for u, v in nx_graph.edges() if u == v))
            stats['num_connected_components'].append(nx.number_connected_components(G_undir))
            stats['avg_degree'].append(graph.num_edges / graph.num_nodes)  # Assuming directed graph
            stats['validated'].append(graph.validate(raise_on_error=True)) 
            
        return DataFrame(stats)

    def download(self):
        api_stem = 'https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/'
        for file in grid_files:
            api_url = f'{api_stem}{file}.tab3.zip'
            file_short = file.split('/')[0]
            file_name = f'{file_short}.tab3.zip'
            print("Trying to access:", api_url)  # Debug print to verify the URL
            file_path = os.path.join(self.raw_dir, file_name)
            if not os.path.exists(file_path):
                response = requests.get(api_url, stream=True)
                if response.status_code == 200:
                    os.makedirs(self.raw_dir, exist_ok=True)
                    makedirs(self.raw_dir)
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192): 
                            # If you have chunk encoded response uncomment if
                            # and set chunk_size parameter to None.
                            #if chunk: 
                            f.write(chunk)
                
                    extract_zip(file_path, self.raw_dir)
                else:
                    print(f"Failed to retrieve data. Status Code: {response.status_code}")
            else:
                print('File already exists')

    def process(self):
        # Process the raw data and save it to the processed directory
        print(self.processed_paths[0])
        if not os.path.exists(self.processed_paths[0]):
            data, years  = read_biogrid_data(self.raw_dir, self.name)
            self.versions = years
            # Apply the functions specified in pre_filter and pre_transform
            if self.pre_filter is not None:           
                data = [d for d in data if self.pre_filter(d)]                
                

            if self.pre_transform is not None:
                data = [self.pre_transform(d) for d in data] 
            self.data = data
            # Store the processed data
            torch.save(self.data, self.processed_paths[0])
            torch.save(self.versions, self.processed_paths[1])