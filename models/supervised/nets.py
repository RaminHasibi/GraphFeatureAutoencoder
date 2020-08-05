
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv
import torch.nn as nn
import torch
from torch.nn import Linear as Lin

from models.supervised.layers import ExpGraphConv

class FAE_ExpGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(FAE_ExpGraphConv, self).__init__()
        self.conv1 = ExpGraphConv(in_channels, 32, 64, aggr='mean')
        self.conv2 = ExpGraphConv(64, 16, 32, aggr='mean')
        self.lin = Lin(32, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.lin(x)

class FAE_SAGEConv(nn.Module):
    def __init__(self , in_channels, out_channels, hidden=16):
        super(FAE_SAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_channels, 64, concat =True)
        self.conv2 = SAGEConv(64, 32, concat= True)
        self.lin = Lin(32, 1)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x

class FAE_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(FAE_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.lin = Lin(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x

class FAE_GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(FAE_GraphConv, self).__init__()
        self.conv1 = GraphConv(in_channels, 64,aggr='mean')
        self.conv2 = GraphConv(64, 32, aggr='mean')
        self.lin = Lin(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return x



class AE_MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(AE_MLP, self).__init__()
        self.lin1 = Lin(in_channels, 64)
        self.lin2 = Lin(64, 32)
        self.lin3 = Lin(32, 1)

    def forward(self, data):
        x = data.x
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        return self.lin3(x)
