import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 64, cached=True)
        self.conv2 = GCNConv(64, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class Embedding_ExpGAE(GAE):
    def __init__(self, in_channels, out_channels):
        encoder = Encoder(in_channels, out_channels)
        super(Embedding_ExpGAE, self).__init__(encoder=encoder)
        self.predictor_lr = LinearRegression()
        self.predictor_rf = RandomForestRegressor(n_estimators=20, max_depth=2)
    def fit_predictor(self, z, y):
        self.predictor_lr.fit(z, y)
        self.predictor_rf.fit(z, y)

    def predict(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.predictor_lr.predict(z.cpu().data.numpy()), self.predictor_rf.predict(z.cpu().data.numpy())
