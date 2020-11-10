
import pprint as pp


import torch
import numpy as np

from options import get_options

from utils.functions import load_data_class, load_model
from eval import supervised_prediction_eval, imputation_eval, embedding_prediction_eval
from imputer import impute


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Choose the dataset to use
    data_class = load_data_class(opts.dataset)

    # Load data from load_path
    data = data_class(root=opts.datadir, network=opts.network)[0]
        
    
    # Preprocess node features
    if opts.no_features:
        print('node ids used')
        data.x = torch.eye(data.num_nodes).to(opts.device)
    
    data = data.to(opts.device)
    model_class = load_model(opts)
    assert opts.problem in ['Prediction', 'Imputation', 'Imputation_eval'], 'only support prediction or imputation of expression values'
    print(data)
    if opts.problem == 'Prediction':
        if not opts.embedding:
            supervised_prediction_eval(model_class, data, opts)
        else:
            embedding_prediction_eval(model_class, data, opts)
    
    elif opts.problem == 'Imputation_eval':
        imputation_eval(model_class, data, opts)
    elif opts.problem == 'Imputation':
        imputed = impute(model_class, data, opts)
        np.save(opts.model + opts.network + '_imputed.npy', imputed.cpu().detach().numpy())
    


if __name__ == "__main__":
    run(get_options())