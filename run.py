
import os
import json
import pprint as pp


import torch
from sklearn.preprocessing import StandardScaler

from options import get_options

from utils.functions import load_data_class, load_model
from eval import prediction_eval, imputation_eval
from imputer import impute


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Choose the dataset to use
    data_class = load_data_class(opts.dataset)

    # Load data from load_path
    data = data_class(root=opts.datadir, network=opts.network)[0]

    # Preprocess node features
    if not opts.features:
        data.x = torch.eye(data.num_nodes)


    model_class = load_model(opts.model)
    assert opts.problem in ['Prediction', 'Imputation'], 'only support prediction or imputation of expression values'

    if opts.problem == 'Prediction':
        prediction_eval(model_class, data, opts)
    elif opts.problem == 'Imputation_eval':
        imputation_eval(model_class, data, opts)
    elif opts.problem == 'Imputation':
        impute(model_class, data, opts)


if __name__ == "__main__":
    run(get_options())