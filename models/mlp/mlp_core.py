"""MLP Baseline."""

import os
import random
import time

import numpy as np
import torch
import tqdm
from torch import nn
from sklearn.metrics import r2_score

from models.vae.base import BaseDetector
from models.torch_utils import LinearBlock, get_optimizer_by_name, get_criterion_by_name
from utils import save_pickle, load_pickle


class RandomizedWindowDataset(torch.utils.data.Dataset):

    def __init__(self, X, reward, treatment_id, window, dtype=torch.float32, tid_dtype=torch.int64):
        self.n_patients, self.n_time_steps = reward.shape
        self.reward = torch.as_tensor(reward, dtype=dtype)
        self.treatment_id = torch.as_tensor(treatment_id, dtype=tid_dtype)
        time_steps = np.arange(0, X.shape[0], self.n_time_steps)
        X = np.concatenate([np.expand_dims(X[i:i+self.n_time_steps, :], 0) for i in time_steps], axis=0) # [Patient, time, features]
        self.X = torch.as_tensor(X, dtype=dtype)
        self.window = window
        self.dtype = dtype
        self.tid_dtype = tid_dtype

    def __len__(self):
        return self.n_patients * self.n_time_steps

    def idx_to_2d(self, idx):
        """idx \in [0, n_patients*n_time_steps] -> (patient_id, time_step)"""
        patient_id = idx // self.n_time_steps
        time_step = idx % self.n_time_steps
        return (patient_id, time_step)

    def _twindow(self, idx):
        lower = max(0, idx-self.window)
        upper = min(self.n_time_steps, idx+self.window)
        if upper - lower < 2 * self.window:
            # if the window is too small, we need to pad the time steps
            if lower == 0:
                upper = lower + 2*self.window
            else:
                lower = upper - 2*self.window
        return np.arange(lower, upper, 1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        p, t = self.idx_to_2d(idx)
        twindow = self._twindow(t)
        return (self.X[p, twindow, :], self.reward[p, twindow], self.treatment_id[p, twindow])


class MLPModel(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_list,
                 average_means,
                 hidden_activation_name='relu',
                 output_activation_name='sigmoid',
                 batch_norm=False, dropout_rate=0.2):
        super(MLPModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_list = hidden_list
        self.output_dim = output_dim
        self.average_means = average_means
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.featurizer = self._build_featurizer()
        self.predictor = LinearBlock(self.hidden_list[-1], self.output_dim,
                                     activation_name=self.output_activation_name,
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

    def forward(self, x_time, treatment_id):
        batch_size, time_step, feature_dim = x_time.shape
        x = self.featurizer(x_time)
        if self.average_means:
            mu = torch.mean(x, axis=1)
            potential_outcomes = self.predictor(mu).unsqueeze(1).repeat(1, time_step, 1)
        else:
            potential_outcomes = self.predictor(x)
        reward = torch.gather(potential_outcomes, -1, treatment_id.unsqueeze(-1)).squeeze()
        return reward

    def get_potential_outcomes(self, x_time):
        """Only used in evaluation."""
        batch_size, time_step, feature_dim = x_time.shape
        x = self.featurizer(x_time)
        if self.average_means:
            mu = torch.mean(x, axis=1)
            potential_outcomes = self.predictor(mu).unsqueeze(1).repeat(1, time_step, 1)
        else:
            potential_outcomes = self.predictor(x)
        return potential_outcomes


class MLP(BaseDetector):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_list,
                 average_means,
                 hidden_activation_name='relu',
                 output_activation_name='sigmoid',
                 batch_norm=False,
                 dropout_rate=0,
                 lr=1e-3,
                 epoch_num=10,
                 batch_size=32,
                 criterion_name='mse',
                 device=None, random_state=42,
                 use_compile=False, compile_mode='default',
                 optimizer_name='adam',
                 optimizer_params: dict = {'weight_decay': 1e-5},
                 criterion_params: dict = {},
                 verbose=1,
                 **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_list = hidden_list
        self.average_means = average_means
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.criterion_name = criterion_name
        self.optimizer_name = optimizer_name
        self.device = device
        self.random_state = random_state
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.verbose = verbose
        self.optimizer_params = optimizer_params
        self.criterion_params = criterion_params

        self.n = kwargs.get('n', 1)
        self.l = kwargs.get('l', 1)
        self.criterion = get_criterion_by_name(criterion_name, **criterion_params)

        # set random seed for reproducibility
        self._set_seed(self.random_state)

        # decide device based on availablity
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # If you want to use MPS, uncomment the following lines
            # self.device = torch.device(
            #     "mps" if torch.backends.mps.is_available() else self.device)

        self.build_model()

    def build_model(self):
        self.model = MLPModel(input_dim=self.input_dim,
                              output_dim=self.output_dim,
                              hidden_list=self.hidden_list,
                              average_means=self.average_means,
                              hidden_activation_name=self.hidden_activation_name,
                              output_activation_name=self.output_activation_name,
                              batch_norm=self.batch_norm,
                              dropout_rate=self.dropout_rate)

    def fit(self, train_set, val_set, **kwargs):
        """Train the model.

        Parameters
        ----------
        train_set : RandomizedWindowDataset
        val_set : RandomizedWindowDataset
        """
        self.training_prepare()
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        self.history = self.train(train_loader, val_set)
        return self.history

    def training_prepare(self):
        self.model = self.model.to(self.device)
        self.optimizer = get_optimizer_by_name(model=self.model,
                                               name=self.optimizer_name,
                                               lr=self.lr,
                                               **self.optimizer_params)
        if self.use_compile:
            self.model = torch.compile(model=self.model,
                                       mode=self.compile_mode)
            print('Model compiled.')
        self.model.train()

    def training_forward(self, batch_data):
        x, y, treatments = batch_data
        x = x.to(self.device)
        y = y.to(self.device)
        treatments = treatments.to(self.device)
        self.optimizer.zero_grad()
        reward = self.model(x, treatments)
        loss = self.criterion(reward, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, train_loader, val_set):
        """Train the model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The training data loader.
        """
        history = {'loss': [], 'val_loss': [], 'val_r2':[], 'time': [], 'val_time': []}
        for epoch in tqdm.trange(self.epoch_num, desc=f'Training: ', disable=not self.verbose == 1):
            start_time = time.time()
            overall_loss = []
            self.model.train()
            for batch_data in train_loader:
                loss = self.training_forward(batch_data)
                overall_loss.append(loss)
            # loss could be a tuple or a single value
            if isinstance(loss, (tuple, list)):
                overall_loss = np.mean([l for l in overall_loss])
            else:
                overall_loss = np.mean(overall_loss)
            # Recording
            train_time = time.time()
            history['loss'].append(overall_loss)
            history['time'].append(train_time - start_time)

            # Evaluate the model
            self.model.eval()
            with torch.no_grad():
                val_result = self.evaluating_forward((val_set.X, val_set.reward, val_set.treatment_id))
            # Recording
            eval_time = time.time()
            history['val_time'].append(eval_time - train_time)
            history['val_loss'].append(val_result['loss'])
            history['val_r2'].append(val_result['r2'])
        return history

    def eval(self, test_set, **kwargs):
        """Evaluate the model.

        Parameters
        ----------
        test_set : RandomizedWindowDataset
        """
        self.model.to(self.device)
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        eval_result = {'loss': 0, 'r2': 0, 'count': len(dataloader)}
        for batch_data in tqdm.tqdm(dataloader):
            batch_result = self.evaluating_forward(batch_data)
            eval_result['loss'] += batch_result['loss']
            eval_result['r2'] += batch_result['r2']
        eval_result['loss'] /= eval_result['count']
        eval_result['r2'] /= eval_result['count']

        if self.verbose > 0:
            print(f"Avg Loss: {eval_result['loss']}")
            print(f"R2: {eval_result['r2']}")
        return eval_result

    @torch.no_grad()
    def evaluating_forward(self, batch_data):
        x, y, treatments = batch_data
        x = x.to(self.device)
        y = y.to(self.device)
        treatments = treatments.to(self.device)
        reward = self.model(x, treatments)
        loss = self.criterion(reward, y)
        r2 = r2_score(y.cpu().numpy().flatten(), reward.cpu().numpy().flatten())
        return {'loss': loss.item(), 'r2': r2}

    @torch.no_grad()
    def predict(self, x, treatment_id):
        """Used for bandit time."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        treatment_id = torch.as_tensor(treatment_id, dtype=torch.int64, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return self.model(x, treatment_id).cpu().numpy()

    @torch.no_grad()
    def get_potential_outcomes(self, x):
        """Used for bandit time."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return self.model.get_potential_outcomes(x).cpu().numpy()

    def save(self, path):
        """Save the model to specified folder.

        Parameters
        ----------
        path : str
            The folder to save the model.
        """
        os.makedirs(path, exist_ok=True)
        torch_model_path = os.path.join(path, 'torch_model.bin')
        model_path = os.path.join(path, 'model.bin')
        torch.save(self.model.state_dict(), torch_model_path)
        ## Don't pickle the model and the optimizer
        model = self.model
        del self.model
        del self.optimizer
        save_pickle(model_path, self)
        self.model = model

    def load_weights(self, path):
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    @classmethod
    def load(cls, path, device=None):
        """Load the model from the specified folder.

        Parameters
        ----------
        path : str
            The path to load the model.
        device : str, optional (default=None)
            The device to use for the model. If None, it will be decided
            automatically.

        Returns
        -------
        model : BaseDeepLearningDetector
            The loaded model.
        """
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_model_path = os.path.join(path, 'torch_model.bin')
        model_path = os.path.join(path, 'model.bin')
        self = load_pickle(model_path)
        self.device = device
        self.build_model()
        self.model.load_state_dict(torch.load(torch_model_path, map_location=device, weights_only=True))
        return self

    @staticmethod
    def _set_seed(random_state):
        """Set random seed for reproducibility
        """
        os.environ['PYTHONHASHSEED'] = str(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
