import torch

from torch_geometric.data.edge_index import EdgeIndex
from torch_geometric.testing import onlyCUDA, withCUDA


def test_edge_index():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3))
    assert isinstance(adj, EdgeIndex)
    assert str(adj) == ('EdgeIndex([[0, 1, 1, 2],\n'
                        '           [1, 0, 2, 1]])')
    assert adj.sparse_size == (3, 3)
    assert adj.sort_order is None

    assert not isinstance(adj.as_tensor(), EdgeIndex)

    assert not isinstance(adj + 1, EdgeIndex)


def test_edge_index_fill_cache():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row')
    adj.validate().fill_cache()
    assert adj.sparse_size == (3, None)
    assert torch.equal(adj._rowptr, torch.tensor([0, 1, 3, 4]))

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order='col')
    adj.validate().fill_cache()
    assert adj.sparse_size == (None, 3)
    assert torch.equal(adj._colptr, torch.tensor([0, 1, 3, 4]))


@withCUDA
def test_edge_index_to(device):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = adj.to(torch.int)
    assert isinstance(out, EdgeIndex)
    assert out.dtype == torch.int

    out = adj.to(torch.float)
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == torch.float

    out = adj.to(device)
    assert isinstance(out, EdgeIndex)
    assert out.device == device


@onlyCUDA
def test_edge_index_cpu_cuda():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])

    out = adj.cuda()
    assert isinstance(out, EdgeIndex)
    assert out.is_cuda

    out = out.cpu()
    assert isinstance(out, EdgeIndex)
    assert not out.is_cuda


def test_edge_index_cat():
    adj1 = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3))
    adj2 = EdgeIndex([[1, 2, 2, 3], [2, 1, 3, 2]], sparse_size=(4, 4))

    out = torch.cat([adj1, adj2], dim=1)
    assert out.size() == (2, 8)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size == (4, 4)
    assert out.sort_order is None

    out = torch.cat([adj1, adj2], dim=0)
    assert out.size() == (4, 4)
    assert not isinstance(out, EdgeIndex)


def test_edge_index_flip():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order='row')
    adj.fill_cache()

    out = adj.flip(0)
    assert isinstance(out, EdgeIndex)
    assert torch.equal(out, torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]]))
    assert out.sparse_size == (None, 3)
    assert out.sort_order == 'col'
    assert torch.equal(out._colptr, torch.tensor([0, 1, 3, 4]))

    out = adj.flip([0, 1])
    assert isinstance(out, EdgeIndex)
    assert torch.equal(out, torch.tensor([[1, 2, 0, 1], [2, 1, 1, 0]]))
    assert out.sparse_size == (None, 3)
    assert out.sort_order is None
    assert out._colptr is None