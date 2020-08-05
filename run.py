
import os
import json
import pprint as pp


import torch

from options import get_options

from utils.functions import load_data_class


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
    dataset = load_data_class(opts.dataset)[0]

    if opts.features:



    # Load data from load_path
    load_data = dataset(root=opts.datadir, network=opts.network)


if __name__ == "__main__":
    run(get_options())