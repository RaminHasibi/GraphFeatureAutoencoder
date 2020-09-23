import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Graph Feature Auto-Encoder for Prediction of Gene Expression Values")

    # Data
    parser.add_argument('--problem', default='Imputation_eval', help="Want to predict or Impute the dataset")
    parser.add_argument('--network', type=str, default='Genetic')
    parser.add_argument('--dataset', type=str, default='Ecoli')
    parser.add_argument('--datadir', type=str, default='../data/Expression_Values/Ecoli')

    # Model

    parser.add_argument('--model', type=str, default='MLP')
    parser.add_argument('--embedding', action='store_true', help='Whether to make predictions on the graph embedding')
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--out_channels', type=int, default=32)

    # Training

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=float, default=500)
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed to use')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_features', action='store_true',
                        help='Whether to use Expression values as node features or not')
    parser.add_argument('--norm', action='store_false',
                        help='Whether to normalize Expression values or not')
    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_false', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_false', help='Disable progress bar')

    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}_{}".format(opts.problem, opts.model, opts.dataset),
        opts.run_name
    )
    return opts