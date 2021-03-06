import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Graph Feature Auto-Encoder for Prediction of Gene Expression Values")

    # Data
    parser.add_argument('--problem', default='Imputation_eval', help="Want to predict or Impute the dataset "
                               "(Values in ['Prediction', 'Imputation', 'Imputation_eval'])")
    parser.add_argument('--network', type=str, default='MousePPI', help="(Values in ['MousePPI'] for RNA and "
                                                       "['TF_net', 'PPI', 'Genetic'] for Ecoli)")
    parser.add_argument('--dataset', type=str, default='RNA', help="(Values in ['Ecoli', 'RNA'])")
    parser.add_argument('--datadir', type=str, default='../data/Expression_Values/SingleCellRNA/Mouse', 
                       help="Path to data directory")

    # Model

    parser.add_argument('--model', type=str, default='FeatGraphConv', help="Values in ['GraphConv', 'GCN', 'SAGEConv',"
                                                                           "'FeatGraphConv','MLP' ,"
                                                                           " 'Magic', 'LR', 'RF']")
    parser.add_argument('--embedding', action='store_true', help='Whether to make predictions on the graph embedding '
                                                                 '(only in prediction problem)')
    parser.add_argument('--random_graph', action='store_true', help='Whether to make predictions based on random graph structure')
    parser.add_argument('--random_graph_alpha', type=float, default=.021, help="alpha hyper=parameter for generating"
                                                                               "random graph in Prediction problem")
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--out_channels', type=int, default=32)

    # Training

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=float, default=20000)
    parser.add_argument('--seed', type=int, default=12345, help='Random seed to use')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_features', action='store_true',
                        help='Whether to use Expression values as node features or not')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda

    return opts