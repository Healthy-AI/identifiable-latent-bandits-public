import os

import numpy as np

from utils import load_pickle, read_config
from bandits.bandit_utils import get_random_seed
from data.ilb import generate_ILB


class LinearEnv:
    """Environment for a linear bandit.

    Args:
        mean: The mean for the bandit instance. [n]
        theta: The matrix for arm parameters. [n_arms, n]
        reward_sigma: Standart deviation for reward. (Normal distribution)
        random_state: Random seed, only used in action().
    """

    def __init__(self, mean, theta, reward_sigma, reward_seed=None):
        self.theta = theta
        self.mean = mean
        self.reward_sigma = reward_sigma
        self.random = get_random_seed(reward_seed)

    @classmethod
    def init_from_ckpt(self, mean, ckpt_dir, reward_seed=None):
        theta = os.path.join("/".join(ckpt_dir.split("/")[:-1]), "theta_true.bin")
        yaml_path = os.path.join("/".join(ckpt_dir.split("/")[:-4]), "config.yaml")
        theta = load_pickle(theta)
        reward_sigma = read_config(yaml_path).reward_sigma
        return LinearEnv(mean=mean, theta=theta, reward_sigma=reward_sigma, reward_seed=reward_seed)

    @property
    def potentials(self):
        return self.theta @ self.mean

    @property
    def best_arm(self):
        return np.argmax(self.potentials)

    @property
    def max_reward(self):
        return np.max(self.potentials)

    def expected_reward(self, chosen_action):
        return self.theta[chosen_action] @ self.mean

    def action(self, chosen_action):
        """Returns the reward."""
        reward_noise = self.random.normal(loc=0, scale=self.reward_sigma)
        reward = self.expected_reward(chosen_action)
        return reward + reward_noise, self.max_reward - reward

    def regret(self, action_list):
        """Expected regret"""
        return self.max_reward - np.array(
            [self.expected_reward(a) for a in action_list]
        )


class NonLinearEnv:
    """Environment for a linear bandit.

    Args:
        mean: The mean for the bandit instance. [n]
        theta: The matrix for arm parameters. [n_arms, n]
        reward_sigma: Standart deviation for reward. (Normal distribution)
        random_state: Random seed, only used in action().
    """

    def __init__(self, mean, theta, reward_sigma, reward_seed=None):
        self.theta = theta
        self.mean = mean
        self.reward_sigma = reward_sigma
        self.random = get_random_seed(reward_seed)

    @classmethod
    def init_from_ckpt(self, mean, ckpt_dir, reward_seed=None):
        theta = os.path.join("/".join(ckpt_dir.split("/")[:-1]), "theta_true.bin")
        yaml_path = os.path.join("/".join(ckpt_dir.split("/")[:-4]), "config.yaml")
        theta = load_pickle(theta)
        reward_sigma = read_config(yaml_path).reward_sigma
        return NonLinearEnv(mean=mean, theta=theta, reward_sigma=reward_sigma, reward_seed=reward_seed)

    @property
    def potentials(self):
        return self.theta.apply(self.mean)

    @property
    def best_arm(self):
        return np.argmax(self.potentials)

    @property
    def max_reward(self):
        return np.max(self.potentials)

    def expected_reward(self, chosen_action):
        return self.potentials[chosen_action]

    def action(self, chosen_action):
        """Returns the reward."""
        reward_noise = self.random.normal(loc=0, scale=self.reward_sigma)
        reward = self.expected_reward(chosen_action)
        return reward + reward_noise, self.max_reward - reward

    def regret(self, action_list):
        """Expected regret"""
        return self.max_reward - np.array(
            [self.expected_reward(a) for a in action_list]
        )


class ILBGenerator:
    """Bandit for generating ILB data.

    Use LVMInstance if you are using the LVM as well.
    Use the same latent_seed as the LVMInstance.
    """

    def __init__(self, checkpoint_dir, obs_seed, **kwargs):
        self.ckpt_dir = checkpoint_dir
        self.mixing = load_pickle(self._mixing_path)
        self.config = read_config(self._yaml_path)
        self.data_dim = self.mixing.data_dim
        self.latent_noise_sigma = self.config.latent_noise_sigma
        self.random = get_random_seed(obs_seed)
        self.theta = self._get_thetas()
        self.noise_dist = self.config.__dict__.get("latent_noise_dist", "Gaussian") # default

    @property
    def _yaml_path(self):
        parent_dir = "/".join(self.ckpt_dir.split("/")[:-4])
        return os.path.join(parent_dir, "config.yaml")

    @property
    def _mixing_path(self):
        parent_dir_old = "/".join(self.ckpt_dir.split("/")[:-4])
        mixing_path_old = os.path.join(parent_dir_old, "NonLinear.bin")
        parent_dir_new = "/".join(self.ckpt_dir.split("/")[:-1])
        mixing_path_new = os.path.join(parent_dir_new, "NonLinear.bin")
        if os.path.exists(mixing_path_new):
            return mixing_path_new
        elif os.path.exists(mixing_path_old):
            return mixing_path_old
        else:
            raise FileNotFoundError("`NonLinear.bin` has not been found")

    def _get_thetas(self):
        parent_dir = "/".join(self.ckpt_dir.split("/")[:-1])
        theta_path = os.path.join(parent_dir, "theta_true.bin")
        theta = load_pickle(theta_path)
        return theta

    @property
    def X_dim(self):
        return self.data_dim

    @property
    def Z_dim(self):
        return self.data_dim
    
    def refresh(self, instance_id=None):
        pass

    def step(self, patient_mean):
        """
        Returns:
            patient_mean, x_it, z_it, time
        """
        x_it, z_it, _, mu = generate_ILB(data_dim=self.data_dim,
                                        n_patients=1,
                                        len_treatment=1,
                                        random=self.random,
                                        mixing=self.mixing,
                                        sigma_latent=None,
                                        noise_dist=self.noise_dist,
                                        sigma_noise=self.latent_noise_sigma, 
                                        patient_mean=patient_mean) # kwargs
        return patient_mean, x_it.flatten(), z_it.flatten(), None

    def call(self, x, return_ica=False):
        return self.mixing.inverse(x)

    def get_theta(self, return_ica=False):
        return {"W": self.theta, "b": np.zeros(self.theta.shape[0])}

    def close(self):
        pass
