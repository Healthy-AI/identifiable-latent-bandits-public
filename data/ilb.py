import warnings

import numpy as np
from scipy.stats import ortho_group

from utils import read_config

def leaky_ReLU_1d(d, negSlope):
    """
    one dimensional implementation of leaky ReLU
    """
    if d > 0:
        return d
    else:
        return d * negSlope


leaky1d = np.vectorize(leaky_ReLU_1d)


def leaky_ReLU(D, negSlope):
    """
    implementation of leaky ReLU activation function
    """
    assert negSlope > 0  # must be positive
    return leaky1d(D, negSlope)


def leaky_ReLU_1d_inv(d, negSlope):
    """
    one dimensional implementation of inverse leaky ReLU
    """
    if d > 0:
        return d
    else:
        return d / negSlope


leaky1dinv = np.vectorize(leaky_ReLU_1d_inv)


def leaky_ReLU_inv(D, negSlope):
    """
    implementation of leaky ReLU activation function
    """
    assert negSlope > 0  # must be positive
    return leaky1dinv(D, negSlope)


def sigmoidAct(x):
    """
    one dimensional application of sigmoid activation function
    """
    return 1. / (1 + np.exp(-1 * x))


def sigmoidInv(x):
    """
    one dimensional application of inverse sigmoid activation function
    """
    return np.log(x / (1 - x))


class NonInvertibleWarning(Warning):
    pass


def separate_segments(x, len_treament, len_full):
    """Seperate train and validation for each patient.
    Args:
        x: data to be separated
        len_treament: len_treatment
        len_full: len_treatment + len_test
    """
    time_steps = np.arange(0, x.shape[0], len_full)
    data = []
    data_val = []
    for time in time_steps:
        data.append(x[time:time+len_treament])
        data_val.append(x[time+len_treament:time+len_full])
    data = np.concatenate(data, axis=0)
    data_val = np.concatenate(data_val, axis=0)
    return data, data_val


def seperate_train_val_data(x, y, source, len_treatment, len_full):
    x, x_val = separate_segments(x, len_treatment, len_full)
    label, label_val = separate_segments(y, len_treatment, len_full)
    source, source_val = separate_segments(source, len_treatment, len_full)
    return x, label, source, x_val, label_val, source_val


def generate_ILB(data_dim, n_patients, len_treatment, random, mixing, sigma_latent=1., noise_dist='Gaussian', sigma_noise=1., **kwargs):
    Sigma_noise = np.identity(data_dim) * sigma_noise
    label = np.concatenate([np.repeat(i, len_treatment) for i in range(n_patients)])
    # Z_i
    if 'patient_mean' in kwargs:
        mu = kwargs['patient_mean'].reshape(1, -1)
    else:
        Sigma_latent = np.identity(data_dim) * sigma_latent
        mu = random.multivariate_normal(mean=np.zeros(data_dim), cov=Sigma_latent, size=n_patients) # Z_i ~ U
    # Z_it
    if noise_dist == "Gaussian":
        source  = np.concatenate([random.multivariate_normal(mean=z_i, cov=Sigma_noise, size=len_treatment) for z_i in mu])
    elif noise_dist == "Uniform":
        _u = sigma_noise / 2
        source  = np.concatenate([z_i.reshape(1, -1) + random.uniform(-_u, _u, size=(len_treatment, data_dim)) for z_i in mu])
    elif noise_dist == "Laplace":
        source  = np.concatenate([z_i.reshape(1, -1) + random.laplace(0, sigma_noise, size=(len_treatment, data_dim)) for z_i in mu])
    # X_it
    x = np.copy(source)
    x = mixing.apply(x)
    return x, source, label, mu


def generate_ILB_data(data_dim, n_patients, len_treatment, seed, mixin_kwargs, sigma_latent=1., noise_dist='Gaussian', sigma_noise=1., len_val=None):
    random = np.random.RandomState(seed)
    len_treatment = len_treatment if len_val is None else len_treatment + len_val
    mixing = NonLinear(data_dim, **mixin_kwargs)
    x, source, label, mu = generate_ILB(data_dim, n_patients, len_treatment, random, mixing, sigma_latent, noise_dist, sigma_noise)
    if len_val is not None:
        train_val_ = seperate_train_val_data(x, label, source, len_treatment-len_val, len_treatment)
        return (*train_val_, mu, mixing, random)
    return x, label, source, mu, mixing, random


def call_theta_1d(theta, mu, treatment_id=None):
    if isinstance(theta, np.ndarray):
        return theta@mu if treatment_id is None else theta[treatment_id] @ mu
    elif isinstance(theta, NonLinear):
        return theta.apply(mu) if treatment_id is None else theta.apply(mu)[treatment_id]


def generate_rewards(mu, theta, treatments, reward_sigma, random):
    n_patients, len_treatment = treatments.shape
    n_treatment = np.max(treatments) + 1
    rewards = np.zeros_like(treatments, dtype=np.float64)
    potential_outcomes = np.zeros((n_patients, len_treatment, n_treatment), dtype=np.float64)
    for p in range(n_patients):
        for t in range(len_treatment):
            treatment_id = treatments[p, t]
            rewards[p, t] = call_theta_1d(theta, mu[p], treatment_id)
            potential_outcomes[p, t] = call_theta_1d(theta, mu[p])
    noise = random.normal(0, reward_sigma, size=rewards.shape)
    rewards_noise = rewards + noise
    return {'theta': theta, 'treatment': treatments, 'reward': rewards_noise, 'theta_zed': rewards,
            'potential_outcomes': potential_outcomes, 'noise': noise}


def get_theta(random, data_dim, n_treatment, theta_linear):
    theta = random.normal(size=(n_treatment, data_dim))
    theta /= np.linalg.norm(theta, axis=-1, keepdims=True) # theta[i] for the i-th treatment
    if not theta_linear:
        first_layer = theta.T # .T as NonLinear applies from the right
        seed_ = random.randint(42)
        theta = NonLinear(data_dim=n_treatment, n_layers=2, seed=seed_)
        theta.mixingList[0] = first_layer
    return theta


def generate_treatment_history(mu, n_patients, data_dim, n_treatment, len_treatment, reward_sigma, seed, theta_linear=True):
    random = np.random.RandomState(seed)
    theta = random.normal(size=(n_treatment, data_dim))
    theta /= np.linalg.norm(theta, axis=-1, keepdims=True) # theta[i] for the i-th treatment
    if not theta_linear:
        first_layer = theta.T
        seed_ = random.randint(42) # fixed
        theta = NonLinear(data_dim=n_treatment, n_layers=2, seed=seed_)
        theta.mixingList[0] = first_layer
    random_treatments = random.randint(n_treatment, size=(n_patients, len_treatment))
    random_treatments = np.sort(random_treatments, axis=-1)
    return generate_rewards(mu=mu, theta=theta, treatments=random_treatments, reward_sigma=reward_sigma, random=random)


class NonLinear:
    """Invertible MLP w/ leaky ReLU and sigmoid activaition functions.
    Emission function g for ILB.
    """

    def __init__(self, data_dim, n_layers, nonlinear='leaky', relu_slope=0.2, noise=None, seed=None):
        self.data_dim = data_dim
        self.n_layers = n_layers
        self.nonlinear = nonlinear
        self.relu_slope = relu_slope
        self.seed = seed
        self.random = np.random.RandomState(seed)
        self.noise = noise
        self.mixingList = [ortho_group.rvs(self.data_dim, random_state=self.seed) for _ in range(self.n_layers - 1)]
        self.inverse_list = [np.linalg.inv(A) for A in self.mixingList][::-1]

    def _apply_noise(self, x):
        if self.__dict__.get("noise") == None: # legacy
            x_noise = x
        elif isinstance(self.noise, str): #legacy
            x_noise = x + self.random.normal(0, .5, size=x.shape)
        elif self.noise.dist == "Gaussian":
            x_noise = x + self.random.normal(0, self.noise.sigma, size=x.shape)
        elif self.noise.dist == "Uniform":
            _u = self.noise.sigma / 2
            x_noise = x + self.random.uniform(-_u, _u, size=x.shape)
        else:
            raise ModuleNotFoundError("Specify noise  as one of ('Gaussian', 'Uniform', None)!"\
                                      f" {self.noise} is not recognized.")
        return x_noise

    def apply(self, x):
        for l in range(self.n_layers - 1):
            A = self.mixingList[l]

            # we first apply non-linear function, then causal matrix!
            if self.nonlinear == 'leaky':
                x = leaky_ReLU(x, self.relu_slope)
            elif self.nonlinear == 'sigmoid':
                x = sigmoidAct(x)
            # apply mixing:
            x = np.dot(x, A)
        return self._apply_noise(x)

    def inverse(self, z):
        if self.noise is not None:
            txt = f".apply() method has {self.noise} noise. The function is not invertible"
            warnings.warn(txt, NonInvertibleWarning)
        for l in range(self.n_layers - 1):
            A = self.inverse_list[l]
            z = np.dot(z, A)

            if self.nonlinear == 'leaky':
                z = leaky_ReLU_inv(z, self.relu_slope)
            elif self.nonlinear == 'sigmoid':
                z = sigmoidInv(z)
        return z


def estimate_reward_mean_var(treatment_data):   
    """Estimates the variance for each treatment."""
    n_treatment = treatment_data['treatment'].max() + 1
    treatment_lengths = [np.sum(treatment_data['treatment']== i, 1) for i in range(n_treatment)]
    means = []
    variances = []
    for t_lengths in treatment_lengths:
        var = []
        m = []
        for l, patient in zip(t_lengths, treatment_data['reward']):
            var.append(np.var(patient[:l]))
            m.append(patient[:l])
        N = len(var)
        variance_reward_i = np.sum(var) / N
        variances.append(variance_reward_i)
        means.append(np.mean(np.concatenate(m)))
    return {'means': np.array(means), 'variances': np.array(variances)}


def pack_data_dict(x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test, **kwargs):
    data_dict = {
        'x': x,
        'y': y,
        's': s,
        'x_val': x_val,
        'y_val': y_val,
        'source_val': source_val,
        'x_test': x_test,
        'source_test': source_test,
        'mu': mu,
        'mu_test': mu_test,
        'mixing': mixing,
        'treatment_data': treatment_data,
        'treatment_data_val': treatment_data_val,
        'treatment_data_test': treatment_data_test,
        }
    return data_dict


def unpack_data_dict(data_dict):
    x = data_dict['x']
    y = data_dict['y']
    s = data_dict['s']
    x_val = data_dict.get('x_val')
    y_val = data_dict.get('y_val')
    source_val = data_dict.get('source_val')
    mu = data_dict['mu']
    mixing = data_dict['mixing']
    x_test = data_dict['x_test']
    source_test = data_dict['source_test']
    mu_test = data_dict['mu_test']
    treatment_data = data_dict['treatment_data']
    treatment_data_val = data_dict.get('treatment_data_val')
    treatment_data_test = data_dict['treatment_data_test']
    return x, y, s, x_val, y_val, source_val, mu, mixing, x_test, source_test, mu_test, treatment_data, treatment_data_val, treatment_data_test


class ILBDataConfig:
    """Currently not used, generates ILB data from a config file."""

    def __init__(self, config):
        if isinstance(config, str):
            config = read_config(config)
        self.data_dim = config.data_dim
        self.n_patients = config.n_segments
        self.n_treatment = config.n_treatment
        self.n_layers = config.n_layers
        self.n_time_steps = config.n_obs_per_seg
        self.latent_mean = None
        self.latent_sigma = config.latent_sigma
        self.latent_noise_sigma = config.latent_noise_sigma
        self.latent_noise_dist = config.__dict__.get("latent_noise_dist", "Gaussian")
        self.reward_sigma = config.reward_sigma
        self.seed = config.data_seed
        self.emission_noise = config.__dict__.get("emission_noise", None)
        self.theta_linear = config.__dict__.get("theta_linear", True)
        self.mixin_kwargs = {"n_layers": None,
                             "nonlinear": 'leaky',
                             "relu_slope": 0.2,
                             "noise": self.emission_noise,
                             "seed": self.seed,
                             }

    def generate_data(self, n_time_steps, n_layers):
        data_dim = self.data_dim
        n_segments = self.n_patients
        n_treatment = self.n_treatment
        data_seed = self.seed
        latent_sigma = self.latent_sigma
        latent_noise_sigma = self.latent_noise_sigma
        reward_sigma = self.reward_sigma
        mixin_kwargs = self.mixin_kwargs
        mixin_kwargs['n_layers'] = n_layers
        x, y, s, x_val, y_val, source_val, mu, mixing, ilb_random = generate_ILB_data(
                                                                data_dim=data_dim,
                                                                n_patients=n_segments,
                                                                len_treatment=n_time_steps,
                                                                seed=data_seed,
                                                                mixin_kwargs=mixin_kwargs,
                                                                sigma_latent=latent_sigma,
                                                                noise_dist=self.latent_noise_dist,
                                                                sigma_noise=latent_noise_sigma,
                                                                len_val=20)
        x_test, source_test, _, mu_test = generate_ILB(data_dim=data_dim,
                                                       n_patients=50,
                                                       len_treatment=n_time_steps,
                                                       mixing=mixing,
                                                       random=ilb_random,
                                                       sigma_latent=latent_sigma,
                                                       noise_dist=self.latent_noise_dist,
                                                       sigma_noise=latent_noise_sigma)
        treatment_data = generate_treatment_history(mu=mu,
                                                    n_patients=n_segments,
                                                    data_dim=data_dim,
                                                    n_treatment=n_treatment,
                                                    len_treatment=n_time_steps,
                                                    reward_sigma=reward_sigma,
                                                    seed=data_seed,
                                                    theta_linear=self.theta_linear)
        treatment_data_val = generate_treatment_history(mu=mu,
                                                        n_patients=n_segments,
                                                        data_dim=data_dim,
                                                        n_treatment=n_treatment,
                                                        len_treatment=20,
                                                        reward_sigma=reward_sigma,
                                                        seed=data_seed,
                                                        theta_linear=self.theta_linear)
        treatment_data_test = generate_treatment_history(mu=mu_test,
                                                         n_patients=50,
                                                         data_dim=data_dim,
                                                         n_treatment=n_treatment,
                                                         len_treatment=n_time_steps,
                                                         reward_sigma=reward_sigma,
                                                         seed=data_seed,
                                                         theta_linear=self.theta_linear)
        return pack_data_dict(**locals())
