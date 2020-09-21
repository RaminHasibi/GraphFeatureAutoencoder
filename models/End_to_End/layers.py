from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.nn import Linear
import torch


class ExpGraphConv(MessagePassing):
    def __init__(self, in_channels,hidden , out_channels, aggr='mean', bias = True,
                 **kwargs):
        super(ExpGraphConv, self).__init__(aggr=aggr, **kwargs)
        self.lin1 = Linear(2*hidden, out_channels, bias=bias)
        self.lin2 = Linear(in_channels, hidden, bias=bias)
    def forward(self, x, edge_index, edge_weight=None, size=None):
        edge_index,edge_weight = add_remaining_self_loops(edge_index=edge_index,edge_weight = edge_weight)
        h = self.lin2(x)
        return self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, h):
        return self.lin1(torch.cat((h, aggr_out), 1))