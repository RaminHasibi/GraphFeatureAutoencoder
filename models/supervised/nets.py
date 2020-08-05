from models.supervised.layers import ExpGraphConv
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv
import torch.nn as nn
import torch
from torch.nn import  Linear as Lin




class PredictorExpGraphConv(nn.Module):
    def __init__(self , in_channels, out_channels, hidden=16):
        super(PredictorExpGraphConv, self).__init__()
        self.conv1 = ExpGraphConv(in_channels,64, 128, aggr='mean')
        self.conv2 = ExpGraphConv(128,32, 64, aggr='mean')
        self.lin1 = Lin(64, 32)
        self.lin2 = Lin(32, 1)
    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x,edge_index))
        x = torch.relu(self.lin1(x))
        return self.lin2(x)

class PredictorSAGEConv(nn.Module):
    def __init__(self , in_channels, out_channels, hidden=16):
        super(PredictorSAGEConv, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.lin = Lin(out_channels, 1)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv(x, edge_index))
        return self.lin(x)

class PredictorGCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(PredictorGCN, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.lin = Lin(out_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv(x, edge_index))
        return self.lin(x)



class PredictorGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(PredictorGraphConv, self).__init__()
        self.conv = GraphConv(in_channels, out_channels,aggr='mean')
        self.lin = Lin(out_channels, 1)

    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = torch.relu(self.conv(x, edge_index))
        return self.lin(x)



class PredictorMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden=16):
        super(PredictorMLP, self).__init__()
        self.lin1 = Lin(in_channels, 128)
        self.lin2 = Lin(128, 64 )
        self.lin3 = Lin(64, 1)

    def forward(self, data):
        x = data.x
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        return self.lin3(x)