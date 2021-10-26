import argparse

import torch

 

def add_argument():

   

    parser = argparse.ArgumentParser(description='DS')

 

    parser.add_argument('--with_cuda',

                        default=True if torch.cuda.is_available() else False,

                        action='store_true',

                        help='use CPU in case there\'s no GPU support')

    parser.add_argument('--use_ema',

                        default=False,

                        action='store_true',

                        help='whether use exponential moving average')

    parser.add_argument('-b',

                        '--batch_size',

                        default=1,

                        type=int,

                        help='mini-batch size (default: 32)')

    parser.add_argument('-e',

                        '--epochs',

                        default=30,

                        type=int,

                        help='number of total epochs (default: 30)')

    parser.add_argument('--local_rank',

                        type=int,

                        default=-1,

                        help='local rank passed from distributed launcher')

    parser.add_argument('--deepspeed_config',

                        type=str,

                        default='ds_config.json',

                        help='location of config file')

    parser.add_argument('--deepspeed',

                        type=bool,

                        default=True,

                        help='whether to use deepspeed')

   

    args = parser.parse_args([])

 

    return args