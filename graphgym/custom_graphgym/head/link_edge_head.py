import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_head
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config


@register_head('custom')
class ExampleGNNEdgeHead(torch.nn.Module):
    r"""A GNN prediction head for edge-level/link-level prediction tasks.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        # Module to decode edges from node embeddings:
        print('INITIALIZING NEW CUSTOM EDGE HEAD')
        if cfg.model.edge_decoding == 'concat':
            self.layer_post_mp = MLP(
                new_layer_config(
                    dim_in * 2,
                    dim_out,
                    cfg.gnn.layers_post_mp,
                    has_act=False,
                    has_bias=True,
                    cfg=cfg,
                ))
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(f"Binary edge decoding "
                                 f"'{cfg.model.edge_decoding}' is used for "
                                 f"multi-class classification")
            self.layer_post_mp = MLP(
                new_layer_config(
                    dim_in,
                    dim_in,
                    cfg.gnn.layers_post_mp,
                    has_act=False,
                    has_bias=True,
                    cfg=cfg,
                ))
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = torch.nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError(f"Unknown edge decoding "
                                 f"'{cfg.model.edge_decoding}'")

    def _apply_index(self, batch):
        index = f'{batch.split}_edge_index'
        label = f'{batch.split}_edge_label'
        return batch.x[batch[index]], batch[label]

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch) # encode using batch.x batch.edge_index
        pred, label = batch.x[batch.edge_label_index], batch.edge_label # test using only supervision edges
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        return pred, label