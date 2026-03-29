import argparse
import torch
from src.pipeline import run_mamba_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='CUDA training.')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Dimension of representations')
    parser.add_argument('--layer', type=int, default=2,
                        help='Num of layers')
    parser.add_argument('--n-test', type=int, default=365,
                        help='Size of test set')
    parser.add_argument('--ts-code', type=str, default='2330.TW',
                        help='Stock code')  
    parser.add_argument('--seq-len', type=int, default=20,
                        help='size of sliding window')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='size of batch')
    parser.add_argument('--show-plot', action='store_false', default=True)

    args = parser.parse_args()
    params_dict = vars(args)
    params_dict['use_cuda'] = args.use_cuda and torch.cuda.is_available()
    result = run_mamba_pipeline(params_dict)