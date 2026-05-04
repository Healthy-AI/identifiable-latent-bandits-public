import numpy as np

from data.adcb import adcb_preprocess


def parse_ckptdir(ckpt_dir):
    n_layers, n_segment_len, model_seed = ckpt_dir.split('/')[-3:]
    return int(n_layers), int(n_segment_len), int(model_seed)


def handle_observations(obs_result, dftype, minmax=None, add_ptmarry=False):
    if dftype == 'adcb':
        if obs_result is False:
            raise ValueError('No more data to observe')
        rid, x_it, z_it, t = obs_result
        x_it = adcb_preprocess(x_it, minmax=minmax, add_ptmarry=add_ptmarry)
        z_it = np.array(list(z_it.values()))
    elif dftype == 'ilb':
        mean, x_it, z_it, _ = obs_result
    return x_it, z_it


def handle_rewards(reward_result, dftype):
    if dftype == 'ilb':
        reward, regret = reward_result
    elif dftype == 'adcb':
        reward = reward_result['reward']
        regret = reward_result['regret']
    return reward, regret


def get_random_seed(state):
    if isinstance(state, int):
        return np.random.RandomState(state)
    elif isinstance(state, np.random.RandomState):
        return state
    elif state is None:
        # Randomly initialized
        return np.random.RandomState(state)
    else:
        raise TypeError("Random state must be Int or `np.random.RandomState`")


def inference_theta_hat(x, W, b):
    return W @ x.T + np.tile(b, (x.shape[0], 1)).T
