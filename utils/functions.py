

def load_data_class(name):
    from dataset import Ecoli_Exp,RnaSeq
    dataset = {
        'Ecoli': Ecoli_Exp,
        'RNA': RnaSeq
    }.get(name, None)
    assert dataset is not None, "Currently unsupported problem: {}!".format(name)
    return dataset

def load_model(name):
    from models.supervised.nets import PredictorGraphConv, PredictorGCN, PredictorSAGEConv, PredictorExpGraphConv\
        ,PredictorMLP
    model = {'PredictorGraphConv': PredictorGraphConv,
          'PredictorGCN': PredictorGCN,
          'PredictorSAGEConv': PredictorSAGEConv ,
          'PredictorExpGraphConv': PredictorExpGraphConv,
          'PredictorMLP': PredictorMLP
    }.get(name, None)
    assert model is not None, "Currently unsupported model: {}!".format(name)
    return model