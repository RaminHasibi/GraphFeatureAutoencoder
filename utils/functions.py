import torch

def load_data_class(name):
    from dataset import Ecoli_Exp,RnaSeq
    dataset = {
        'Ecoli': Ecoli_Exp,
        'RNA': RnaSeq
    }.get(name, None)
    assert dataset is not None, "Currently unsupported problem: {}!".format(name)
    return dataset

def load_model(name):
    from models.supervised.nets import FAE_GraphConv, FAE_GCN, FAE_SAGEConv, FAE_ExpGraphConv\
        ,AE_MLP
    model = {'GraphConv': FAE_GraphConv,
          'GCN': FAE_GCN,
          'SAGEConv': FAE_SAGEConv ,
          'ExpGraphConv': FAE_ExpGraphConv,
          'MLP': AE_MLP
    }.get(name, None)
    assert model is not None, "Currently unsupported model: {}!".format(name)
    return model


def index_to_mask(index, size):
    # if len(size) == 2
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask