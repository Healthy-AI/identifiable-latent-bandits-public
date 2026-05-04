import os

import numpy as np
import torch

from models.sequence_regressor.train import load_model
from models.preprocessing import pca
from models.tcl.tcl_wrapper_gpu import TCLInference
from models.vae import VAE
from utils import load_pickle, read_config
from bandits.bandit_utils import parse_ckptdir, inference_theta_hat


class AbstractLVMInstance:
    """LVM Environment to run for LVM bandits.

    Use this LVM Environment in bandit models to call the LVM.
    call() to predict the latent state on those observations.
    call_theta() to for the potential outcomes from the latent state.

    Args:
        checkpoint_dir: checkpoint directory where the LVM is stored.
    """

    def __init__(self, checkpoint_dir, return_ica):
        self.ckpt_dir = checkpoint_dir
        self.config = read_config(self._yaml_path)
        self._n_layers, self._n_segment_len, _ = parse_ckptdir(self.ckpt_dir)
        self.ica = load_pickle(os.path.join(self.ckpt_dir, "fast_ica.bin"))
        self.ica._whiten = self.ica.whiten ## For sklearn compat
        self.theta_hat_ica = load_pickle(os.path.join(self.ckpt_dir, "thetas_ica.bin"))
        self.theta_hat = load_pickle(os.path.join(self.ckpt_dir, "thetas.bin"))
        self.return_ica = return_ica

    @property
    def _yaml_path(self):
        parent_dir = "/".join(self.ckpt_dir.split("/")[:-4])
        return os.path.join(parent_dir, "config.yaml")

    def _return_ica(self, return_ica):
        return return_ica if return_ica is not None else self.return_ica

    def _handle_input(self, x):
        pass

    def call_lvm(self, x):
        pass

    def call(self, x, return_ica=None):
        x = self._handle_input(x)
        z_hat = self.call_lvm(x)
        return_ica = self._return_ica(return_ica)
        if return_ica:
            z_hat = self.ica.transform(z_hat)
        return z_hat

    def call_theta(self, z, return_ica=None):
        theta_dict = self.get_theta(return_ica=return_ica)
        return inference_theta_hat(z, **theta_dict).T

    def get_theta(self, return_ica=None):
        return_ica = self._return_ica(return_ica)
        return self.theta_hat_ica if return_ica else self.theta_hat

    def close(self):
        pass


class LVMInstance(AbstractLVMInstance):
    """LVM Environment to run for LVM bandits.

    Use this LVM Environment in bandit models to call the LVM.
    call() to predict the latent state on those observations.
    call_theta() to for the potential outcomes from the latent state.

    Args:
        checkpoint_dir: checkpoint directory where the LVM is stored.
        dftype: dataset type 'adcb' or 'ilb'
    """

    def __init__(self, checkpoint_dir, return_ica, dftype, **kwargs):
        super().__init__(checkpoint_dir, return_ica=return_ica)
        self.dftype = dftype
        self._is_csv = dftype == "adcb"
        self.list_hidden = [self.config.data_dim * 2] * (self._n_layers - 1) + [self.config.data_dim]
        self.inference_tcl = TCLInference(list_hidden_nodes=self.list_hidden, ckpt_dir=self.ckpt_dir)
        if not self._is_csv:
            self.pca_params = load_pickle(os.path.join(self.ckpt_dir, "pca_params.bin"))

    @property
    def latent_dim(self):
        return self.config.data_dim

    def _handle_input(self, x):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] == self.config.data_dim:
            x = x.T
        return x

    def call_lvm(self, x):
        x = x if self._is_csv else self.preprocess(x)
        z_hat = self.inference_tcl(x)
        return z_hat

    def preprocess(self, x):
        x, _ = pca(x, num_comp=self.latent_dim, params=self.pca_params)
        return x

    def close(self):
        self.inference_tcl.close()


class VAEInstance(AbstractLVMInstance):
    """VAE Environment to run for LVM bandits.

    Use this LVM Environment in bandit models to call the LVM.
    call() to predict the latent state on those observations.
    call_theta() to for the potential outcomes from the latent state.

    Args:
        checkpoint_dir: checkpoint directory where the LVM is stored.
    """

    def __init__(self, checkpoint_dir, return_ica, **kwargs):
        super().__init__(checkpoint_dir, return_ica=return_ica)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE.load(self.ckpt_dir, device=self.device)
        self.model.evaluating_prepare()

    def _handle_input(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.model.preprocess:
            sensor = x.T
            assert sensor.ndim == 2, 'x has to have a dim of 2' 
            sensor, _ = pca(sensor, num_comp=sensor.shape[0], params=self.model.pca_parm)
            x = sensor.T
        return torch.as_tensor(x, dtype=torch.float32).to(self.device)

    def call_lvm(self, x):
        return self.model.predict(x).cpu().numpy()


class SequenceModelInstance:

    def __init__(self, checkpoint_dir, **kwargs):
        self.actions = []
        self.rewards = []
        self.history = {"potential_outcomes": []}
        self.ckpt_dir = checkpoint_dir
        self.config = read_config(self._yaml_path)
        self._n_layers, self._n_segment_len, _ = parse_ckptdir(self.ckpt_dir)
        self.model = load_model(self.ckpt_dir, self.config.data_dim, self.config.n_treatment, self._n_layers)
        if os.path.exists(self._minmax_path):
            self.minmax = load_pickle(self._minmax_path)

    def _reset(self):
        self.actions = []
        self.rewards = []
        self.history = {"potential_outcomes": []}

    @property
    def _yaml_path(self):
        parent_dir = "/".join(self.ckpt_dir.split("/")[:-4])
        return os.path.join(parent_dir, "config.yaml")

    @property
    def _minmax_path(self):
        return os.path.join(self.ckpt_dir, "minmax.bin")

    def call(self, x):
        """x: [1, bandit_time, data_dim]."""
        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        potential_outcomes = self.model.get_potential_outcomes(x).detach().numpy()
        self.history['potential_outcomes'].append(potential_outcomes[0, -1, :])
        return potential_outcomes

    def choose_action(self, X):
        return np.argmax(self.call(X)[0, -1, :])

    def update(self, chosen_action, reward, *args, **kwargs):
        self.actions.append(chosen_action)
        self.rewards.append(reward)
