import numpy as np
import pandas as pd

from bandits.environment import Environment
from data.adcb import ADCB_CONFIG
from utils import read_config


class ADCBGym(Environment):
    def __init__(self, data_path, reward_sigma, reward_seed, **kwargs):
        """
        Initializes the environment

        args
        data_path: str
            path to the data
        """
        self.data = pd.read_csv(data_path)
        config = read_config(ADCB_CONFIG)
        self.outcome_columns = config.potoutcome_cols
        self.noise_col = [config.noise_col]
        self.Z_cols = config.z_cols
        self.X_cols = ["PTMARRY"] + config.x_cols # Always includes PTMARRY
        self.t_col = config.t_col
        self.id_col = [config.id_col]
        self._xdim = len(self.X_cols) + len(config.ptmarry_cols) if config.add_ptmarry else len(self.X_cols) - 1
        self.buffer_ = None
        self.reward_seed = reward_seed
        self.random = np.random.RandomState(reward_seed)
        self.reward_sigma = reward_sigma

    @property
    def all_instances(self):
        return self.data[self.id_col[0]].unique()

    @property
    def X_dim(self):
        return self._xdim

    @property
    def Z_dim(self):
        return len(self.Z_cols)

    def refresh(self, instance_id):
        self.get_new_instance(instance_id)

    def get_new_instance(self, rid):
        """
        Resets the environment and returns the selected patient that can be stepped through during inference
        """
        print("Resetting environment...")
        self.done = False
        self.d = None
        self.cumulative_rewards, self.cumulative_regrets = 0, 0
        self.tt = 0  # time step returned for context
        self.buffer_ = self.data.groupby(self.id_col).get_group(rid)

    def step(self, instance_id=None):
        """
        Args:
            instance_id: Redundant, only pass None.
        Returns:
            id (dict): patient id
            X (dict): patient features
            Z (dict): patient latent features
            t (dict): time step
        """
        if self.buffer_.shape[0] < 1:
            #self.reset()
            self.done = True
            return False

        self.d = self.buffer_.iloc[0:1]
        self.buffer_ = self.buffer_.iloc[1:]

        assert self.d[self.t_col].values[0] == self.tt
        self.tt += 1

        return (
            self.d[self.id_col].squeeze(),
            {
                key: value[0]
                for key, value in self.d[self.X_cols].to_dict(orient="list").items()
            },
            {
                key: value[0]
                for key, value in self.d[self.Z_cols].to_dict(orient="list").items()
            },
            self.d[self.t_col].squeeze(),
        )

    def action(self, action):
        """Plays an action, returns a reward.
        Returns:
            reward (float): reward
            regret (float): regret
            expected_rewards (float): expected rewards
            expected_regrets (float): expected regrets
            outcomes (dict): counterfactual patient outcomes
        """
        assert self.d[self.t_col].values[0] == self.tt - 1
        outcomes = self.d[self.outcome_columns]
        noise = self.random.normal(0, self.reward_sigma)
        outcomes = np.array(
            [outcomes["Y_" + str(a)].values[0] for a in range(8)], dtype="float64"
        )

        rewards = np.array([(outcomes[a]) for a in range(8)])
        r = rewards[action]
        regret = np.max(rewards) - rewards[action]

        self.cumulative_rewards += r
        self.cumulative_regrets += regret

        return {
            "reward": r + noise,
            "regret": regret, # expected_regret
            "cumulative_rewards": self.cumulative_rewards,
            "cumulative_regrets": self.cumulative_regrets,
            "outcomes": outcomes,
        }
