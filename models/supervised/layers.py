from torch_geometric.nn import GraphConv
from torch_geometric.utils import add_remaining_self_loops
import torch


class ExpGraphConv(GraphConv):
    def __init__(self, in_channels,hidden , out_channels, aggr='mean', bias = True,
                 **kwargs):
        super(ExpGraphConv, self).__init__(in_channels, hidden, aggr=aggr, **kwargs)
        self.lin = torch.nn.Linear(hidden+in_channels, out_channels, bias=bias)
    def forward(self, x, edge_index, edge_weight=None, size=None):
        edge_index,edge_weight = add_remaining_self_loops(edge_index=edge_index,edge_weight = edge_weight)
        h = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, x):
        return self.lin(torch.cat((x, aggr_out), 1))