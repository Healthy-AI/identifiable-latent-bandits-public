import argparse
import os
import pickle

import numpy as np
import torch
import yaml

from runners.simulation_runner import run_ilb_exp, run_vae_exp, run_LSTM_exp
from runners.simulation_runner import eval_ilb_exp, eval_vae_exp, eval_LSTM_exp, run_mlp_exp, eval_mlp_exp
from runners import nonlinear_runner as nl_run
from runners.csv_runner import run_ilb_csv, run_vae_csv, run_LSTM_csv, run_mlp_csv
from runners.csv_runner import eval_ilb_csv, eval_vae_csv, eval_LSTM_csv, eval_mlp_csv
from utils import write_json_file, write_text_file, dict2namespace

RUN_FUNCTIONS = {'ilb': {'ilb': [run_ilb_exp, eval_ilb_exp],
                         'vae': [run_vae_exp, eval_vae_exp],
                         'regression': [run_LSTM_exp, eval_LSTM_exp],
                         'mlp': [run_mlp_exp, eval_mlp_exp],
                         },
                 'adcb': {'ilb': [run_ilb_csv, eval_ilb_csv],
                          'vae': [run_vae_csv, eval_vae_csv],
                          'regression': [run_LSTM_csv, eval_LSTM_csv],
                          'mlp':[run_mlp_csv, eval_mlp_csv],
                          },
                'nonlinear':{'ilb': [nl_run.run_ilb_exp, nl_run.eval_ilb_exp],
                             'vae': [nl_run.run_vae_exp, nl_run.eval_vae_exp],
                             'regression': [nl_run.run_LSTM_exp, nl_run.eval_LSTM_exp],
                             }
                }


def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='ILB', help='Dataset to run experiments.')
    parser.add_argument('--method', type=str, default='ILB', help='Method to employ.')
    parser.add_argument('--config', type=str, default='lvm-ilb.yaml', help='Path to the config file')
    parser.add_argument('--run', type=str, default='run/', help='Path for saving running related data.')
    parser.add_argument('--nSims', type=int, default=10, help='Number of simulations to run')
    parser.add_argument('--start_seed', type=int, default=0, help='Start seed')
    return parser.parse_args()


def make_dirs_simulations(args):
    os.makedirs(args.run, exist_ok=True)
    args.checkpoints = os.path.join(args.run, 'checkpoints', args.method)
    os.makedirs(args.checkpoints, exist_ok=True)


if __name__ == '__main__':
    args = parse_sim()
    print('Running {} experiments using {}'.format(args.dataset, args.method))
    # make checkpoint and log folders
    make_dirs_simulations(args)
    np.random.seed(123)

    if args.dataset.lower() in ['ilb', 'adcb', 'nonlinear']:
        # Read the config and save a copy of it in the checkpoints folder
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.load(f, Loader=yaml.loader.Loader)
        write_text_file(yaml.safe_dump(config), os.path.join(args.checkpoints, 'config.yaml')) # make a copy of the config 
        new_config = dict2namespace(config)
        new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Run the experiment
        print("Start Seed: ", args.start_seed)
        job_type = 'train' if new_config.EVAL is False else 'eval'
        _idx = 1 if new_config.EVAL else 0
        try:
            result = RUN_FUNCTIONS[args.dataset.lower()][args.method.lower()][_idx](args, new_config)
        except Exception as e:
            print(f"Unexpected method should be one of {list(RUN_FUNCTIONS[args.dataset.lower()].keys())} for dataset {args.dataset.lower()}.")
            raise e

        # Save the results
        fname = os.path.join(args.checkpoints, args.method + 'res_' + args.dataset + 'exp_' + str(args.nSims) + '.p')
        pickle.dump(result, open(fname, "wb"))
        write_json_file(result, fname[:-2] + '.json', **{'indent': 4})
    else:
        raise ValueError('Unsupported dataset {}'.format(args.dataset))