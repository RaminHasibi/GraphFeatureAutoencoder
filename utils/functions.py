import torch

def load_data_class(name):
    from dataset import Ecoli_Exp,RnaSeq
    dataset = {
        'Ecoli': Ecoli_Exp,
        'RNA': RnaSeq
    }.get(name, None)
    assert dataset is not None, "Currently unsupported problem: {}!".format(name)
    return dataset

def load_model(opts):
    from models.End_to_End.nets import FAE_GraphConv, FAE_GCN, FAE_SAGEConv, FAE_FeatGraphConv\
        , AE_MLP
    from models.Embedding.model import Embedding_ExpGAE
    from magic import MAGIC
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    if not opts.embedding:
        model = {'GraphConv': FAE_GraphConv,
              'GCN': FAE_GCN,
              'SAGEConv': FAE_SAGEConv ,
              'FeatGraphConv': FAE_FeatGraphConv,
              'MLP': AE_MLP,
              'Magic': MAGIC,
                 'LR': LinearRegression,
                 'RF': RandomForestRegressor
        }.get(opts.model, None)
    else:
        model = Embedding_ExpGAE


    assert model is not None, "Currently unsupported model: {}!".format(opts.model)
    return model


def index_to_mask(index, size):
    
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask