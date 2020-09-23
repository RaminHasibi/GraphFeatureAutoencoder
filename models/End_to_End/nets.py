
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv
import torch.nn as nn
import torch
from torch.nn import Linear as Lin

from models.End_to_End.layers import ExpGraphConv


class FAE_ExpGraphConv(nn.Module):
    def __init__(self, in_channels, opts):
        super(FAE_ExpGraphConv, self).__init__()
        self.opts = opts
        if self.opts.problem == 'Prediction':
            self.conv1 = ExpGraphConv(in_channels, 64, 64, aggr='mean')
            self.conv2 = ExpGraphConv(64, 32, 32, aggr='mean')
            self.lin = Lin(32, 1)
        else:
            self.conv1 = ExpGraphConv(in_channels, 64, 64, aggr='mean')
            self.lin = Lin(64, in_channels)
    def forward(self, data):
        if self.opts.problem == 'Prediction':
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            return self.lin(x)
        else:
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = self.lin(x)
            return x


class FAE_SAGEConv(nn.Module):

    def __init__(self , in_channels, opts):
        super(FAE_SAGEConv, self).__init__()
        self.opts = opts
        if self.opts.problem == 'Prediction':
            self.conv1 = SAGEConv(in_channels, 64)
            self.conv2 = SAGEConv(64, 32)
            self.lin = Lin(32, 1)
        else:
            self.conv1 = SAGEConv(in_channels, 64)
            self.lin = Lin(64, in_channels)
    def forward(self, data):
        if self.opts.problem == 'Prediction':
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = self.lin(x)
            return x
        else:
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = self.lin(x)
            return x


class FAE_GCN(nn.Module):
    def __init__(self, in_channels, opts):
        super(FAE_GCN, self).__init__()
        self.opts = opts
        if self.opts.problem == 'Prediction':
            self.conv1 = GCNConv(in_channels, 64)
            self.conv2 = GCNConv(64, 32)
            self.lin = Lin(32, 1)
        else:
            self.conv1 = GCNConv(in_channels, 64)
            # self.conv2 = GCNConv(64, 32)
            self.lin = Lin(64, in_channels)

    def forward(self, data):
        if self.opts.problem == 'Prediction':
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = self.lin(x)
            return x
        else:
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            # x = torch.relu(self.conv2(x, edge_index))
            x = self.lin(x)
            return x


class FAE_GraphConv(nn.Module):
    def __init__(self, in_channels, opts):
        super(FAE_GraphConv, self).__init__()
        self.opts = opts
        if self.opts.problem == 'Prediction':
            self.conv1 = GraphConv(in_channels, 64, aggr='mean')
            self.conv2 = GraphConv(64, 32, aggr='mean')
            self.lin = Lin(32, 1)
        else:
            self.conv1 = GraphConv(in_channels, 64, aggr='mean')
            # self.conv2 = GraphConv(64, 32,aggr='mean')
            self.lin = Lin(64, in_channels)

    def forward(self, data):
        if self.opts.problem == 'Prediction':
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.relu(self.conv2(x, edge_index))
            x = self.lin(x)
            return x
        else:
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            # x = torch.relu(self.conv2(x, edge_index))
            x = self.lin(x)
            return x


class AE_MLP(nn.Module):
    def __init__(self, in_channels, opts):
        super(AE_MLP, self).__init__()
        self.opts = opts
        if self.opts.problem == 'Prediction':
            self.lin1 = Lin(in_channels, 64)
            self.lin2 = Lin(64, 32)
            self.lin3 = Lin(32, 1)
        else:
            self.lin1 = Lin(in_channels, 64)
            self.lin2 = Lin(64, in_channels)

    def forward(self, data):
        if self.opts.problem == 'Prediction':
            x = data.x
            x = torch.relu(self.lin1(x))
            x = torch.relu(self.lin2(x))
            return self.lin3(x)
        else:
            x = data.x
            x = torch.relu(self.lin1(x))
            x = self.lin2(x)
            return x
