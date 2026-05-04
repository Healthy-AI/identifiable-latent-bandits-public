import argparse
import os
from typing import Union

import numpy as np
import tensorflow.compat.v1 as tf
import torch
import yaml

from utils import save_pickle, dict2namespace, join_list_of_dict, load_pickle, read_config
from data.adcb import ADCB_CONFIG
from data.ilb import estimate_reward_mean_var
from bandits import * # __all__ import

tf.logging.set_verbosity(0)
RewardEnv = Union[LinearEnv, ADCBGym]
ObsEnv = Union[ILBGenerator, ADCBGym]
InferenceModel = Union[ILBGenerator, LVMInstance, VAEInstance]

ADD_PTMARRY = read_config(ADCB_CONFIG).add_ptmarry
LVM_BANDITS = {'greedy1', 'greedy2', 'projected_thompson', 'mab_prior'}
BANDIT_ALGORITHMS = {
    'mab': MAB,
    'greedy1': BayesOptimalGreedy, # Greedy1 v2.
    'greedy2': GreedyBandit2,
    'projected_thompson': ProjectedThompson,
    'regression': SequenceModelInstance,
    'random_feature_bandit': RFB,
    'mab_prior': MABPrior,
    'linear_thompson': LinUCB_v2,
    'warmstart_thompson': WarmStartLinearThompsonSampler,
}


def parse_sim():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='configs/new-bandit.yaml', help='Path to the config file')
    parser.add_argument('--start_seed', type=int, default=0, help='Start Seed')
    parser.add_argument('--end_seed', type=int, default=1, help='End Seed')
    args_parsed = parser.parse_args()
    args = read_bandit_config(args_parsed.config)
    args.start_seed = args_parsed.start_seed
    args.end_seed = args_parsed.end_seed
    return args


def read_bandit_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.loader.Loader)
    config['bandits'] = join_list_of_dict(config['bandits'])
    new_config = dict2namespace(config)
    new_config.bandit_config = join_list_of_dict(new_config.bandit_config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return new_config


def read_lvm_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.loader.Loader)
    new_config = dict2namespace(config)
    new_config.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return new_config


def make_bandit_dirs(args):
    bandit_paths = {}
    banditdir = os.path.join(args.lvm_path, 'bandit_results')
    os.makedirs(banditdir, exist_ok=True)
    for bandit, in_use in args.bandits.__dict__.items():
        if in_use:
            if bandit in LVM_BANDITS and args.lvm == 'oracle':
                bandit = f'{bandit}_oracle'
            bandit_paths[bandit] = os.path.join(banditdir, bandit)
            os.makedirs(bandit_paths[bandit], exist_ok=True)
        else:
            continue
    return bandit_paths


def env_summary(env: RewardEnv):
    if isinstance(env, LinearEnv):
        summary = {'bandit_potentials': list(env.potentials), 
                   'best_arm': int(env.best_arm),
                   'max_reward': float(env.max_reward),
                   }
    elif isinstance(env, ADCBGym):
        exp_rewards = env.buffer_.mean()[env.outcome_columns].to_numpy()
        summary = {'bandit_potentials': list(exp_rewards),
                   'best_arm': int(exp_rewards.argmax()),
                   'max_reward': float(exp_rewards.max()),
                   }
    else:
        raise ValueError(f'Unknown Environment {env}')
    return summary


def run_bandit(bandit, mean_or_rid, n_iter, dftype, obs_env, reward_env=None,
                minmax=None, obs_seed=None, reward_seed=None, is_lvm_bandit=False):
    if reward_env is None:
        reward_env = obs_env
    summary = env_summary(reward_env)
    X = np.empty(shape=(0, obs_env.X_dim)) # Observed data
    Z = np.empty(shape=(0, obs_env.Z_dim)) # Real latents
    cnt = 0
    regret_list = []
    while cnt < n_iter:
        # Run the LVM
        x_it, z_it = handle_observations(obs_env.step(mean_or_rid), dftype=dftype, minmax=minmax, add_ptmarry=ADD_PTMARRY)
        X = np.vstack((X, x_it))
        Z = np.vstack((Z, z_it))
        # Action
        chosen_action = bandit.choose_action(X)
        reward = reward_env.action(chosen_action)
        reward, regret = handle_rewards(reward, dftype=dftype)
        bandit.update(chosen_action=chosen_action, reward=reward, Z=X[-1], x_t=X[-1])
        regret_list.append(regret)
        cnt += 1

    if is_lvm_bandit:
        bandit.lvm.close() # Closes the TF1 Session
    return {'mean': mean_or_rid,
            'obs_seed': obs_seed,
            'reward_seed': reward_seed,
            'history': bandit.history,
            'actions': np.array(bandit.actions).flatten(),
            'rewards': np.array(bandit.rewards).flatten(),
            'regret': np.array(regret_list),
            'cum_regret': np.cumsum(regret_list),
            'summary': summary,
            'X': X,
            'Z': Z,
            'Z_hat': bandit.__dict__.get("Z_hat", None),
            }

if  __name__ == '__main__':
    from pathlib import Path
    args = parse_sim()
    bandit_paths = make_bandit_dirs(args)
    lvm_config = read_lvm_config(Path(args.lvm_path).parents[3] / 'config.yaml')
    DATA_DIM = lvm_config.data_dim
    N_ARMS = lvm_config.n_treatment
    np.random.seed(123)
    minmax = load_pickle(os.path.join(args.lvm_path, 'minmax.bin')) if args.dftype == 'adcb' else None

    def get_new_instance(dftype):
        """Returns observation and reward environments."""
        if dftype == 'ilb':
            Sigma_latent = np.identity(DATA_DIM) * lvm_config.latent_sigma
            mean = RANDOM_STATE.multivariate_normal(mean=np.full(DATA_DIM, 0.), cov=Sigma_latent, size=1).flatten()
            assert OBS_SEED is not None and REWARD_SEED is not None
            obs_env = ILBGenerator(checkpoint_dir=args.lvm_path, obs_seed=OBS_SEED)
            reward_env = LinearEnv.init_from_ckpt(mean=mean, ckpt_dir=args.lvm_path, reward_seed=REWARD_SEED)
            return obs_env, reward_env
        elif dftype == 'adcb':
            env = ADCBGym(args.filepath, reward_sigma=lvm_config.reward_sigma, reward_seed=REWARD_SEED)
            return env, None


    def get_bandit(bandit_name):
        """Returns the bandit instance."""
        if bandit_name not in LVM_BANDITS:
            inference_model = None
        elif args.lvm == 'lvm':
            inference_model = LVMInstance(checkpoint_dir=args.lvm_path, return_ica=args.return_ica, dftype=args.dftype)
        elif args.lvm == 'vae':
            inference_model = VAEInstance(checkpoint_dir=args.lvm_path, return_ica=args.return_ica, dftype=args.dftype)
        elif args.lvm == 'oracle':
            if args.dftype == 'adcb': raise ValueError('Oracle not supporrted for ADCB')
            inference_model = ILBGenerator(checkpoint_dir=args.lvm_path, obs_seed=OBS_SEED)
        else:
            raise ValueError(f'Unknown LVM {args.lvm}')
        #reward_mv = estimate_reward_mean_var(load_pickle(os.path.join(Path(args.lvm_path).parent, 'data_dict.bin'))['treatment_data'])
        kwargs = {
            "lvm": inference_model,
            "n_arms": N_ARMS,
            "seed": seed,
            "reward_sigma": lvm_config.__dict__.get("reward_sigma"),
            #"reward_mean": reward_mv['means'],
            #"reward_variance": reward_mv['variances'],
            **args.bandit_config[bandit_name],
            }
        return BANDIT_ALGORITHMS[bandit_name](**kwargs)

    for seed in range(args.start_seed, args.end_seed):
        print(f'Starting for SEED {seed}')
        instance_id = seed # Seed either decides the random state or indexes the data
        RANDOM_STATE = np.random.RandomState(seed) # Seed for the data generator
        OBS_SEED = 9_000 + seed if args.dftype == 'ilb' else None
        REWARD_SEED = 10_000 + seed if args.dftype == 'ilb' else None
        obs_env, reward_env = get_new_instance(dftype=args.dftype)

        for bandit_name, in_use in  args.bandits.__dict__.items():
            if not in_use:
                continue
            # Continiue if the result already exists
            # Good for re-runs after timeouts
            if bandit_name in LVM_BANDITS and args.lvm == 'oracle':
                model_name = f'{bandit_name}_oracle'
                save_path = os.path.join(bandit_paths[model_name], f'seed_{seed}{model_name}.bin')
            else:
                model_name = f"_{args.lvm.upper()}" if bandit_name in LVM_BANDITS else ''
                save_path = os.path.join(bandit_paths[bandit_name], f'seed_{seed}{model_name}.bin')
            if os.path.exists(save_path):
                continue

            print(f'Running {bandit_name}')
            mean_or_rid = reward_env.mean if args.dftype == 'ilb' else obs_env.all_instances[instance_id]
            obs_env.refresh(mean_or_rid) # quick fix
            result = run_bandit(bandit=get_bandit(bandit_name),
                                mean_or_rid=mean_or_rid,
                                n_iter=args.n_iter,
                                dftype=args.dftype,
                                obs_env=obs_env,
                                reward_env=reward_env,
                                minmax=minmax,
                                obs_seed=OBS_SEED, # Just for name keeping no-function
                                reward_seed=REWARD_SEED, # Just for name keeping no-function
                                is_lvm_bandit= bandit_name in LVM_BANDITS
                                )
            model_name = f"_{args.lvm.upper()}" if bandit_name in LVM_BANDITS else ''
            if bandit_name in LVM_BANDITS and args.lvm == 'oracle':
                bandit_name = f'{bandit_name}_oracle'
                model_name = '_oracle'
            save_pickle(os.path.join(bandit_paths[bandit_name], f'seed_{seed}{model_name}.bin'), result)
