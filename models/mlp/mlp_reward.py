import os

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch import nn

from models.torch_utils import LinearBlock, get_optimizer_by_name, get_criterion_by_name
from runners.plot_utils import plot_loss_and_r2

_ACTIVATION_FN = 'leaky_relu'

def _hidden_layers(data_dim, n_treatment):
    return [data_dim*2, n_treatment, n_treatment*2]


def estimate_rewards(model, mu, treatment):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
    treatment = torch.as_tensor(treatment, dtype=torch.int64, device=device)
    model.to(device)
    with torch.no_grad():
        reward_hat = model(mu, treatment)
    return reward_hat.cpu().numpy()


def estimate_potential_outcomes(model, mu, treatment):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
    treatment = torch.as_tensor(treatment, dtype=torch.int64, device=device)
    model.to(device)
    with torch.no_grad():
        reward_hat = model.get_potential_outcomes(mu)
    return reward_hat.cpu().numpy()


def eval_reward_model(model, mu, treatment_data):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
    treatment = torch.as_tensor(treatment_data['treatment'], dtype=torch.int64, device=device)
    model.to(device)
    with torch.no_grad():
        r_pred = model(mu, treatment)
        val_r2 = r2_score(y_true=treatment_data['reward'].flatten(), y_pred=r_pred.cpu().numpy().flatten())
    return val_r2


def load_reward_model(data_dim, n_treatment, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLPModel(input_dim=data_dim,
                    output_dim=n_treatment,
                    hidden_activation_name=_ACTIVATION_FN,
                    hidden_list=_hidden_layers(data_dim, n_treatment),
                    dropout_rate=0.)
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    return model


class MLPModel(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_list,
                 hidden_activation_name='leaky_relu',
                 #output_activation_name='relu',
                 batch_norm=False, dropout_rate=0.):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_list = hidden_list
        self.output_dim = output_dim
        self.hidden_activation_name = hidden_activation_name
        #self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.featurizer = self._build_featurizer()
        self.predictor = LinearBlock(self.hidden_list[-1], self.output_dim, has_act=False,
                                     #activation_name=self.output_activation_name,
                                     batch_norm=False, dropout_rate=0)

    def _build_featurizer(self):
        decoder_layers = []
        last_neuron_size = self.input_dim
        for neuron_size in self.hidden_list:
            decoder_layers.append(LinearBlock(last_neuron_size, neuron_size,
                                              activation_name=self.hidden_activation_name,
                                              batch_norm=self.batch_norm,
                                              dropout_rate=self.dropout_rate))
            last_neuron_size = neuron_size
        return nn.Sequential(*decoder_layers)

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, mu, treatment_id):
        batch_size, feature_dim = mu.shape
        feats = self.featurizer(mu)
        potential_outcomes = self.predictor(feats)
        reward = torch.gather(potential_outcomes, 1, treatment_id)
        return reward

    @torch.no_grad()
    def get_potential_outcomes(self, mu):
        """Only used in evaluation."""
        if not torch.is_tensor(mu):
            mu = torch.as_tensor(mu)
        mu = mu.to(torch.float32).to(self.get_device())
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
        batch_size, feature_dim = mu.shape
        feats = self.featurizer(mu)
        potential_outcomes = self.predictor(feats)
        return potential_outcomes


def train_reward_model(mu, treatment_data, treatment_data_val, save_path='./theta_model.bin'):
    # Data Prep
    mu = torch.as_tensor(mu, dtype=torch.float32)
    rewards = torch.as_tensor(treatment_data['reward'], dtype=torch.float32)
    treatments = torch.as_tensor(treatment_data['treatment'], dtype=torch.int64)
    rewards_val = torch.as_tensor(treatment_data_val['reward'], dtype=torch.float32)
    treatments_val = torch.as_tensor(treatment_data_val['treatment'], dtype=torch.int64)
    # Constants
    n_patients, data_dim = mu.shape
    n_treatment = torch.max(treatments) + 1
    # Model
    model = MLPModel(input_dim=data_dim,
                    output_dim=n_treatment,
                    hidden_activation_name=_ACTIVATION_FN,
                    hidden_list=_hidden_layers(data_dim, n_treatment),
                    dropout_rate=0.)
    # Prepare Training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mu = mu.to(device)
    rewards = rewards.to(device)
    treatments = treatments.to(device)
    rewards_val = rewards_val.to(device)
    treatments_val = treatments_val.to(device)
    model.to(device)
    train_loader = torch.utils.data.DataLoader((treatments, rewards),
                                               batch_size=32,
                                               shuffle=False,
                                               num_workers=0)
    criterion = get_criterion_by_name('mse', **{})
    optimizer = get_optimizer_by_name(model=model, name='adam', lr=1e-2, **{'weight_decay': 1e-3})
    history = {'loss': [], 'val_loss': [], 'val_r2':[], 'time': [], 'val_time': []}

    for epoch in range(800):
        overall_loss = []
        model.train()
        for batch_data in train_loader:
            t, r_true = batch_data
            t = t.to(torch.int64)
            optimizer.zero_grad()
            r_pred = model(mu, t)
            loss = criterion(r_true, r_pred)
            loss.backward()
            optimizer.step()
            overall_loss.append(loss.item())
        history['loss'].append(np.mean(overall_loss))
        # Eval
        model.eval()
        with torch.no_grad():
            r_val_pred = model(mu, treatments_val)
            val_loss = criterion(r_val_pred, rewards_val)
            val_r2 = r2_score(y_true=rewards_val.cpu().numpy().flatten(), y_pred=r_val_pred.cpu().numpy().flatten())
        history['val_loss'].append(val_loss.item())
        history['val_r2'].append(val_r2)

    # Best possible R2 for sanity check
    best_r2 = r2_score(y_true=rewards_val.cpu().numpy().flatten(), y_pred=treatment_data_val['theta_zed'].flatten())
    history['best_r2'] = best_r2
    # Save model
    ckpt_dir = os.path.dirname(save_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    plot_loss_and_r2(history, ckpt_dir)
    torch.save(model.state_dict(), save_path)
    return history
