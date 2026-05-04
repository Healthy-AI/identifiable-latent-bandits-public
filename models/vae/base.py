# -*- coding: utf-8 -*-
# Author: Tiankai Yang <tiankaiy@usc.edu>
# License: BSD 2 clause
"""Base class for deep learning models."""

import abc
import os
import random
import time
import warnings
from inspect import isfunction
from collections import defaultdict
from inspect import signature

import numpy as np
import torch
import tqdm
from sklearn.utils import check_array

from models.preprocessing import pca
from models.torch_utils import get_optimizer_by_name, get_criterion_by_name
from utils import save_pickle, load_pickle


class BaseDetector(metaclass=abc.ABCMeta):
    """Abstract class."""

    # noinspection PyIncorrectDocstring
    @abc.abstractmethod
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # noinspection PyPep8
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Returns
        -------
        self : object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        # noinspection PyPep8
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name), ),)


def _pprint(params, offset=0, printer=repr):
    # noinspection PyPep8
    """Pretty print the dictionary 'params'

    See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
    and sklearn/base.py for more information.

    :param params: The dictionary to pretty print
    :type params: dict

    :param offset: The offset in characters to add at the begin of each line.
    :type offset: int

    :param printer: The function to convert entries to strings, typically
        the builtin str or repr
    :type printer: callable

    :return: None
    """

    # Do a multi-line justified repr:
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset // 2) * ' '
    for i, (k, v) in enumerate(sorted(params.items())):
        if type(v) is float:
            # use str for representing floating point numbers
            # this way we get consistent representation across
            # architectures and versions.
            this_repr = '%s=%s' % (k, str(v))
        else:
            # use repr of the rest
            this_repr = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if this_line_length + len(this_repr) >= 75 or '\n' in this_repr:
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    # Strip trailing space to avoid nightmare in doctests
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, X, X_dtype=torch.float32):
        self.X = X
        self.X_dtype = X_dtype

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.tensor(self.X[idx, :], dtype=self.X_dtype)


class BaseDeepLearningDetector(BaseDetector):
    """Abstract class for all deep learning models.

    Parameters
    ----------
    lr : float, optional (default=1e-3)
        The learning rate for the optimizer.

    epoch_num : int, optional (default=10)
        The number of epochs to train the model.

    batch_size : int, optional (default=32)
        The batch size for training the model.

    optimizer_name : str, optional (default='adam')
        The name of optimizer used to train the model.
        Available optimizers: 'adam', 'sgd'.

    loss_func : str, optional (default=None)
        The loss function used to train the model.

    criterion : torch.nn.modules, optional (default=None)
        The (customized) loss class inherited from torch.nn.modules.
        Applicable when loss_func is None.

    criterion_name : str, optional (default='mse')
        The name of the criterion used to train the model.
        Available criteria: 'mse', 'mae', 'bce'(binary classification).
        Applicable when loss_func and criterion are None.

    device : str, optional (default=None)
        The device to use for the model. If None, it will be decided
        automatically. If you want to use MPS, set it to 'mps'.

    random_state : int, optional (default=42)
        The random seed for reproducibility.

    preprocess : bool, optional (default=False)
        If True, will apply pca to the input data.

    use_compile : bool, optional (default=False)
        Whether to compile the model.
        If True, the model will be compiled before training.
        This is only available for
        PyTorch version >= 2.0.0. and Python < 3.12.

    compile_mode : str, optional (default='default')
        The mode to compile the model.
        Can be either “default”, “reduce-overhead”,
        “max-autotune” or “max-autotune-no-cudagraphs”.
        See https://pytorch.org/docs/stable/generated/torch.compile.html#torch-compile for details.

    verbose : int, optional (default=1)
        Verbosity mode.
        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

    optimizer_params : dict, optional (default=None)
        Additional parameters for the optimizer.
        For example, `optimizer_params={'weight_decay': 1e-4}`.

    criterion_params : dict, optional (default=None)
        Additional parameters for the criterion.
        For example, `criterion_params={'reduction': 'sum'}`.
    """

    def __init__(self,
                 lr=1e-3, epoch_num=10, batch_size=32,
                 optimizer_name='adam',
                 loss_func=None, criterion=None, criterion_name='mse',
                 device=None, random_state=42, preprocesss=False,
                 use_compile=False, compile_mode='default',
                 verbose=1,
                 optimizer_params: dict = {},
                 criterion_params: dict = {},
                 **kwargs):
        self.lr = lr
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.device = device
        self.random_state = random_state
        self.preprocess = preprocesss
        self.use_compile = use_compile
        self.compile_mode = compile_mode
        self.verbose = verbose
        self.optimizer_params = optimizer_params
        self.criterion_params = criterion_params

        self.n = kwargs.get('n', 1)
        self.l = kwargs.get('l', 1)
        self.data_num = None
        self.feature_size = None

        # set loss function or criterion
        if isfunction(loss_func):
            self.criterion = loss_func
        elif loss_func is not None:
            raise ValueError('Invalid loss function.')
        else:
            if isinstance(criterion, torch.nn.Module):
                self.criterion = criterion
            elif criterion is not None:
                raise ValueError('Invalid criterion class.')
            else:
                if isinstance(criterion_name, str):
                    self.criterion = get_criterion_by_name(name=criterion_name,
                                                           **self.criterion_params)
                else:
                    raise ValueError('Invalid criterion name.')

        # set random seed for reproducibility
        self._set_seed(self.random_state)

        # decide device based on availablity
        if self.device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            # If you want to use MPS, uncomment the following lines
            # self.device = torch.device(
            #     "mps" if torch.backends.mps.is_available() else self.device)

    def fit(self, X, x_test, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        x_test : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of input samples. Not used in unsupervised methods.
        Returns
        -------
        history : dictionary of metrics
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        x_test = check_array(x_test)

        if self.preprocess:
            sensor = X.T
            sensor_test = x_test.T
            assert sensor.ndim == 2, 'x has to have a dim of 2' 
            sensor, self.pca_parm = pca(sensor, num_comp=sensor.shape[0])
            sensor_test, _ = pca(sensor_test, num_comp=sensor.shape[0], params=self.pca_parm)
            X = sensor.T
            x_test = sensor_test.T

        self.data_num, self.feature_size = X.shape
        self.build_model()
        self.training_prepare()
        train_set = TorchDataset(X=X, X_dtype=torch.float32)

        # create data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=self.batch_size,
            shuffle=True, drop_last=True)
        test_set = torch.as_tensor(x_test, dtype=torch.float32).to(self.device)
        # train the model
        history = self.train(train_loader, test_set)
        return history

    def training_prepare(self):
        self.model = self.model.to(self.device)

        # set optimizer
        self.optimizer = get_optimizer_by_name(model=self.model,
                                               name=self.optimizer_name,
                                               lr=self.lr,
                                               **self.optimizer_params)

        if self.use_compile:
            self.model = torch.compile(model=self.model,
                                       mode=self.compile_mode)
            print('Model compiled.')

        self.model.train()

    def early_stopping(self):
        """Early stopping to terminate training when validation loss is not improving.

        Parameters
        ----------
        history : dict
            The history containing 'val_loss' key.

        Returns
        -------
        stop : bool
            Whether to stop training.
        """
        patience = 5
        if len(self.history['val_loss']) > patience:
            recent_losses = self.history['val_loss'][-patience:]
            if all(recent_losses[i] >= recent_losses[i - 1] for i in range(1, patience)):
                print("Early stopping triggered at epoch: ", len(self.history['val_loss']))
                if self.verbose == 2:
                        print(f"final_val_loss={recent_losses[-1]:.4f}")
                return True
        return False

    def train(self, train_loader, test_set):
        """Train the deep learning model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The data loader for training the model.
        Returns
        -------
        history : dictionary of metrics
        """
        self.history = {'loss': [], 'val_loss': []}
        for epoch in tqdm.trange(self.epoch_num,
                                 desc=f'Training: ',
                                 disable=not self.verbose == 1):
            start_time = time.time()
            overall_loss = []
            self.model.train()
            for batch_data in train_loader:
                loss = self.training_forward(batch_data)
                overall_loss.append(loss)
                self.history['loss'].append(loss)
            # loss could be a tuple or a single value
            if isinstance(loss, (tuple, list)):
                overall_loss = np.mean([l for l in overall_loss])
            else:
                overall_loss = np.mean(overall_loss)

            ## Evaluate the model
            self.model.eval()
            with torch.no_grad():
                val_loss, _, _ = self.evaluating_forward(test_set)
                val_loss = val_loss.item()
            self.history['val_loss'].append(val_loss)

            # loss could be a tuple or a single value
            if self.verbose == 2:
                if isinstance(loss, (tuple, list)):
                    print(f'Epoch {epoch + 1}/{self.epoch_num},', end=' ')
                    for i, l in enumerate(loss):
                        print(f'loss_{i}={l:.4f}', end=', ')
                    print(f'time={time.time() - start_time:.2f}s')
                else:
                    print(f'Epoch {epoch + 1}/{self.epoch_num}, '
                          f'loss={overall_loss:.4f}, '
                          f'time={time.time() - start_time:.2f}s')
            if self.early_stopping():
                break
        return self.history

    def evaluate(self, X, batch_size=None):
        """Get latents and reconstruction loss.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.
        batch_size : int, optional (default=None)
            The batch size for processing the input samples.
            If not specified, the default batch size is used.
        Returns
        -------
        latents : numpy array of shape (n_samples, latent_dim)
        """
        X = check_array(X)
        if self.preprocess:
            sensor = X.T
            assert sensor.ndim == 2, 'x has to have a dim of 2' 
            sensor, _ = pca(sensor, num_comp=sensor.shape[0], params=self.pca_parm)
            X = sensor.T
        dataset = TorchDataset(X=X, X_dtype=torch.float32)

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            shuffle=False, drop_last=False)
        latents, scores = self.evaluate_run(data_loader)
        return latents, scores

    def evaluating_prepare(self):
        self.model.to(self.device)
        self.model.eval()

    def evaluate_run(self, data_loader):
        """Evaluate the deep learning model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The data loader for evaluating the model.

        Returns
        -------
        latents : numpy array of shape (n_samples, latent_dim)
        scores : numpy array of shape (n_samples,)
        """
        self.evaluating_prepare()
        latents = []
        scores = []
        with torch.no_grad():
            for batch_data in data_loader:
                _, score, latent = self.evaluating_forward(batch_data)
                latents.append(latent.cpu().numpy())
                scores.append(score)
        latents = np.concatenate(latents)
        scores = np.concatenate(scores)
        return latents, scores

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
        self.model.load_state_dict(torch.load(path, map_location=self.device))

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
        if self.__dict__.get('preprocess') is None: ## for backward support of preprocess
            self.preprocess = False
        self.device = device
        self.build_model()
        self.model.load_state_dict(torch.load(torch_model_path, map_location=device))
        return self

    @staticmethod
    def _set_seed(random_state):
        """Set random seed for reproducibility
        """
        os.environ['PYTHONHASHSEED'] = str(random_state)
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

    @abc.abstractmethod
    def build_model(self):
        """
        Need to define model in this method.
        self.feature_size is the number of features in the input data.
        """
        pass

    @abc.abstractmethod
    def training_forward(self, batch_data):
        """
        Forward pass for training the model.
        Abstract method to be implemented.

        Parameters
        ----------
        batch_data : tuple
            The batch data for training the model.

        Returns
        -------
        loss : float or tuple of float
            The loss.item of the model, or a tuple of loss.item 
            if there are multiple losses.
        """
        # An example implementation:
        # x = batch_data
        # x = x.to(self.device)
        # # x, y = batch_data
        # # x = x.to(self.device)
        # # y = y.to(self.device)
        # self.optimizer.zero_grad()
        # output = self.model(x)
        # loss = self.criterion(output, x)
        # loss.backward()
        # self.optimizer.step()
        # return loss.item()
        pass

    @abc.abstractmethod
    def evaluating_forward(self, batch_data):
        """
        Forward pass for evaluating the model.
        Abstract method to be implemented.

        Parameters
        ----------
        batch_data : tuple
            The batch data for evaluating the model.

        Returns
        -------
        output : numpy array
            The output of the model.
        """
        # An example implementation:
        # x = batch_data
        # x_gpu = x.to(self.device)
        # # x, y = batch_data
        # # x_gpu = x.to(self.device)
        # # y = y.to(self.device)
        # output = self.model(x_gpu)
        # return pairwise_distances_no_broadcast(x.numpy(),
        #                                        output.cpu().numpy())
        pass
