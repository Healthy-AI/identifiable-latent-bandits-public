"""Implementation of bandit algorithms."""

import os 

import numpy as np
from scipy import optimize
from typing import Optional, List

from bandits.bandit_utils import get_random_seed, inference_theta_hat
from models.mlp.mlp_reward import load_reward_model


class MAB:
    """Linear Thompson Sampling."""

    def __init__(self, n_arms, reward_sigma, seed=None, **kwargs):
        self.n_arms = n_arms
        self.reward_sigma = reward_sigma
        self.seed = seed
        self.random = get_random_seed(seed)
        self.actions = np.array([], dtype=np.int32)
        self.rewards = np.array([], dtype=np.float64)
        self.history = self.init_history()
        # initialize
        self.arms = {}
        for i in range(n_arms):
            p = self.get_prior(i)
            self.arms[i] = {"mean": p["mean"][0], "var": p["var"][0]}

    def get_prior(self, arm=None):
        return {"mean": [0.0], "var": [1.0]}

    def init_history(self):
        return {i: self.get_prior(i) for i in range(self.n_arms)}

    @property
    def get_params(self):
        return {
            "mean": [arm["mean"] for _, arm in self.arms.items()],
            "var": [arm["var"] for _, arm in self.arms.items()],
        }

    def _reset(self):
        self.actions = np.array([], dtype=np.int32)
        self.rewards = np.array([], dtype=np.float64)
        self.arms = self.init_history()

    def sample(self):
        """Sample from all arms."""
        return [
            self.random.normal(loc=arm["mean"], scale=np.sqrt(arm["var"]))
            for _, arm in self.arms.items()
        ]

    def update(self, chosen_action, reward, **kwargs):
        """Update TS."""
        self.actions = np.append(self.actions, chosen_action)
        self.rewards = np.append(self.rewards, reward)
        reward_history = self.rewards[self.actions == chosen_action]
        n = len(reward_history)
        prior = self.get_prior(chosen_action)
        pmean, pvar = prior["mean"][0], prior["var"][0]
        var = np.divide(1.0, 1.0 / pvar + n / self.reward_sigma**2)
        mean = var * (pmean / pvar + np.sum(reward_history) / self.reward_sigma**2)
        self.arms[chosen_action]["mean"] = mean
        self.arms[chosen_action]["var"] = var
        self.history[chosen_action]["mean"].append(mean)
        self.history[chosen_action]["var"].append(var)

    def choose_action(self, *args, **kwargs):
        return np.argmax(self.sample())


class LinearThompsonSampler:
    """Linear Thompson Sampling for shared context."""

    def __init__(self, n_arms, d, lambda_, delta, T, R, seed=None, **kwargs):
        """
        :param n_arms: Number of arms.
        :param d: Dimensionality of the shared context vector.
        :param lambda_: Regularization parameter in the prior.
        :param delta: Confidence parameter (not directly used here).
        :param T: Time horizon (used for exploration scaling factor).
        :param R: R^2 is the variance of the measurement noise.
        :param seed: Random seed for reproducibility.
        """
        self.n_arms = n_arms
        self.d = d
        self.lambda_ = lambda_
        self.delta = delta
        self.T = T
        self.R = R
        self.seed = seed
        self.random = np.random.default_rng(seed)

        # Initialize posterior parameters for each arm
        self.V = {i: lambda_ * np.eye(d) for i in range(n_arms)}  # Precision matrix
        self.b = {i: np.zeros(d) for i in range(n_arms)}  # Reward-weighted context
        self.history = self.init_history()
        self.actions = []
        self.rewards = []

    def init_history(self):
        """Initialize history to track the evolution of parameters."""
        return {
            "V": {i: [self.V[i].copy()] for i in range(self.n_arms)},
            "b": {i: [self.b[i].copy()] for i in range(self.n_arms)},
        }

    def get_prior(self, arm=None):
        """Get prior parameters for consistency."""
        return {"mean": [0.0], "var": [1.0]}

    @property
    def get_params(self):
        """Return current posterior mean for each arm."""
        return {i: np.linalg.solve(self.V[i], self.b[i]) for i in range(self.n_arms)}

    def _reset(self):
        """Reset the sampler to its initial state."""
        self.V = {i: self.lambda_ * np.eye(self.d) for i in range(self.n_arms)}
        self.b = {i: np.zeros(self.d) for i in range(self.n_arms)}
        self.history = self.init_history()
        self.actions = []
        self.rewards = []

    def sample(self, X_t):
        """Sample expected reward for each arm."""
        sampled_rewards = []
        for i in range(self.n_arms):
            # Compute posterior mean and covariance
            theta_hat = np.linalg.solve(self.V[i], self.b[i])
            L = np.linalg.cholesky(np.linalg.inv(self.V[i]))  # Covariance
            eta = self.random.normal(size=self.d)

            # Sample from posterior
            theta_t_i = theta_hat + self.beta() * (L @ eta)
            sampled_rewards.append(X_t @ theta_t_i)
        return sampled_rewards

    def update(self, arm, X_t, reward, **kwargs):
        """Update posterior parameters for the chosen arm."""
        self.V[arm] += np.outer(X_t, X_t)
        self.b[arm] += X_t * reward
        self.actions.append(arm)
        self.rewards.append(reward)

        # Update history
        self.history["V"][arm].append(self.V[arm].copy())
        self.history["b"][arm].append(self.b[arm].copy())

    def choose_action(self, X_t):
        """Choose the arm with the highest sampled reward."""
        sampled_rewards = self.sample(X_t)
        return np.argmax(sampled_rewards)

    def beta(self):
        """Compute exploration scale factor."""
        det_part = sum(np.linalg.slogdet(self.V[i])[1] for i in range(self.n_arms)) / 2
        term1 = 2 * (det_part - self.n_arms * np.log(self.lambda_))
        term2 = self.d * np.log(1 + self.T)
        return self.R * np.sqrt(term1 + term2)


class WarmStartLinearThompsonSampler(LinearThompsonSampler):
    """Warm-start Linear Thompson Sampling with shared context."""

    def __init__(self, n_arms, d, alpha, delta, T, R, seed=None, **kwargs):
        """
        :param n_arms: Number of arms.
        :param d: Dimensionality of the shared context vector.
        :param alpha: Regularization parameter for warm-start.
        :param delta: Confidence parameter (not directly used here).
        :param T: Time horizon (used for exploration scaling factor).
        :param R: R^2 is the variance of the measurement noise.
        :param seed: Random seed for reproducibility.
        """
        super().__init__(n_arms, d, lambda_=1, delta=delta, T=T, R=R, seed=seed)
        self.alpha = alpha
        self.mu_prior = {i: np.zeros(d) for i in range(n_arms)}

    def initialize_prior_from_logs(self, X_log, A_log, R_log):
        """Use log data to initialize the prior."""
        for arm in range(self.n_arms):
            arm_indices = np.where(A_log == arm)[0]
            X_arm = X_log[arm_indices]
            R_arm = R_log[arm_indices]

            if len(X_arm) > 0:
                XTX = X_arm.T @ X_arm
                XTy = X_arm.T @ R_arm

                # Compute ridge regression terms
                self.V[arm] = XTX + self.alpha * np.eye(self.d)
                self.b[arm] = XTy

    def _reset(self):
        """Reset to the warm-start state."""
        super()._reset()
        self.mu_prior = {i: np.zeros(self.d) for i in range(self.n_arms)}


class AbstractLVMBandit:
    """Abstract base class for bandits using an LVM.

    Args:
        lvm: LVM path
    """

    def __init__(self, lvm, **kwargs):
        self.actions = []
        self.rewards = []
        self.history = {"mu_hat": [], "reward_estimate": []}
        self.Z_hat = None
        self.lvm = lvm
        self.theta_matrix = self.lvm.get_theta()

    def append_Z_hat(self, z):
        if self.Z_hat is None:
            self.Z_hat = np.empty(shape=(0, z.flatten().shape[0]))
        self.Z_hat = np.vstack((self.Z_hat, z))

    def _reset(self):
        # Bandit Params
        self.actions = []
        self.rewards = []
        self.Z_hat = np.empty(shape=(0, self.data_dim))
        self.history = {"mu_hat": [], "reward_estimate": []}

    def choose_action(self, X):
        pass

    def update(self, chosen_action, reward, *args, **kwargs):
        self.actions.append(chosen_action)
        self.rewards.append(reward)


class GreedyNonLinear(AbstractLVMBandit):
    """Sit and wait."""

    def __init__(self, lvm, **kwargs):
        self.actions = []
        self.rewards = []
        self.history = {"mu_hat": [], "reward_estimate": []}
        self.Z_hat = None
        self.lvm = lvm
        _n = 'theta_model_ica.bin' if self.lvm.return_ica else 'theta_model.bin'
        self._p = os.path.join(self.lvm.ckpt_dir, _n)
        self.reward_model = load_reward_model(5, 20, self._p)

    def choose_action(self, X):
        z_hat = self.lvm.call(X[-1])
        self.append_Z_hat(z_hat)
        mu_hat = np.mean(self.Z_hat, axis=0)  # Best guess for true latent
        R_potential = self.reward_model.get_potential_outcomes(mu_hat)
        self.history["reward_estimate"].append(R_potential)
        return np.argmax(R_potential)


class BayesOptimalGreedy(AbstractLVMBandit):
    """Greedy1 v2."""

    def choose_action(self, X):
        z_hat = self.lvm.call(X[-1])
        self.append_Z_hat(z_hat)
        R_potential_hist = inference_theta_hat(
            self.Z_hat, self.theta_matrix["W"], self.theta_matrix["b"]
        )
        R_potential = np.mean(R_potential_hist, axis=-1)
        self.history["reward_estimate"].append(R_potential)
        return np.argmax(R_potential)


def _join_W_b(W, b):
    return np.hstack((b.reshape(-1, 1), W))


def get_theta_history(actions, theta):
    return np.array([_join_W_b(W=theta['W'], b=theta['b'])[i] for i in actions])


def loss(z, rewards, theta_history, z_mean):
    """
    Args:
        z: Best candidate for z
        rewards: Reward history
        theta_history: [n_dim, history] matrix of theta_hat for chosen actions.
        z_mean: empirical mean of f^-1(x) seen so far.
    """
    mean_term = np.sum(np.square(z - z_mean))
    reward_term = np.mean(np.square(rewards - np.dot(theta_history, z))) # Mean over history
    return mean_term + reward_term


class GreedyBandit2(AbstractLVMBandit):
    """Sit and wait."""

    def choose_action(self, X):
        # Choose Action
        z_hat = self.lvm.call(X[-1])
        self.append_Z_hat(z_hat)
        mu0 = np.mean(self.Z_hat, axis=0)  # (data_dim,)
        mu0 = np.hstack((1., mu0))
        theta_history = get_theta_history(
            actions=self.actions, theta=self.theta_matrix
        )  # W: (cnt_bandit_iter, data_dim), b:(cnt_bandit_iter)
        if len(self.rewards) > 0:
            result = optimize.minimize(
                fun=loss,
                x0=mu0,
                args=(np.array(self.rewards), theta_history, mu0),
                method="L-BFGS-B",
                bounds=None,
            )
            mu_hat = result["x"]
        else:
            mu_hat = mu0
        self.history["mu_hat"].append(mu_hat)
        W, b = self.theta_matrix["W"], self.theta_matrix["b"]
        R_potential = _join_W_b(W=W, b=b) @ mu_hat
        self.history["reward_estimate"].append(R_potential)
        return np.argmax(R_potential)


class ProjectedThompson(AbstractLVMBandit):
    """
    Univariate and Multivariate updates on a single posterior.
    For the variance and expected values of the rewards, observe that:
    - \E[R|A] = \E[\theta_A^T Z]
    - Var[R|A] = Var[\theta_A^T Z] + Var[\e_R] = \theta_A^T \Sigma \theta_A + \sigma_R^2 (*)
               = \sigma^2 \theta_A^T \theta_A + \sigma_R^2  (for \Sigma = \sigma^2 I)

    For (*) see:
    (https://stats.stackexchange.com/questions/504181/projection-of-a-multivariate-normal-into-a-univariate-normal-calculating-varian)

    Args:
        See AbstractLVMBandit.
    """

    def __init__(self, lvm, seed=None, **kwargs):
        super().__init__(lvm=lvm)
        self.seed = seed
        self.random = get_random_seed(seed)

    def sample(self, reward_estimate):
        t, dim = self.Z_hat.shape
        history = get_theta_history(actions=self.actions, theta=self.theta_matrix)
        th = _join_W_b(**self.theta_matrix)
        V = np.sum([i.reshape(-1,1) @ i.reshape(1,-1) for i in history], 0)
        V = t * np.eye(dim+1) + V
        Sigma = th @ np.linalg.inv(V) @ th.T
        return self.random.multivariate_normal(mean=reward_estimate, cov=Sigma)

    def choose_action(self, X):
        # Choose Action
        z_hat = self.lvm.call(X[-1])
        self.append_Z_hat(z_hat)
        mu0 = np.mean(self.Z_hat, axis=0)  # (data_dim,)
        mu0 = np.hstack((1., mu0))
        theta_history = get_theta_history(
            actions=self.actions, theta=self.theta_matrix
        )  # W: (cnt_bandit_iter, data_dim), b:(cnt_bandit_iter)
        if len(self.rewards) > 0:
            result = optimize.minimize(
                fun=loss,
                x0=mu0,
                args=(np.array(self.rewards), theta_history, mu0),
                method="L-BFGS-B",
                bounds=None,
            )
            mu_hat = result["x"]
        else:
            mu_hat = mu0
        self.history["mu_hat"].append(mu_hat)
        W, b = self.theta_matrix["W"], self.theta_matrix["b"]
        R_potential = _join_W_b(W=W, b=b) @ mu_hat
        self.history["reward_estimate"].append(R_potential)
        return np.argmax(self.sample(R_potential))


class MABPrior(MAB):
    """MAB with prior."""

    def __init__(self, n_arms, reward_sigma, lvm, use_lvm, seed=None, **kwargs):
        super().__init__(n_arms, reward_sigma, seed, **kwargs)
        self.lvm = lvm
        self.use_lvm = use_lvm
        self.theta_matrix = self.lvm.get_theta()
        self.reward_sigma = reward_sigma
        variances = kwargs.get("reward_variance", None)
        self.variances = (
            np.ones(n_arms) * self.reward_sigma
            if np.any(variances)
            else variances
        )
        self.variances = np.ones(n_arms) * 1.
        if not use_lvm:
            self.means = kwargs.get("reward_means", None)
            self.arms = {}
            for i in range(self.n_arms):
                p = self.get_prior(i)
                self.arms[i] = {"mean": p["mean"][0], "var": p["var"][0]}

    def prior(self):
        """Prior variance for the rewards."""
        return self.means, self.variances

    def get_prior(self, arm):
        if not hasattr(self, "means"):  ## Just for compat w/ MAB
            return {"mean": [0.0], "var": [1.0]}
        mean, variance = self.prior()
        return {"mean": [mean[arm]], "var": [variance[arm]]}

    def greedy(self, X):
        z_hat = self.lvm.call(X[-1])
        mu_hat = z_hat.flatten()
        R_potential = self.theta_matrix["W"] @ mu_hat + self.theta_matrix["b"]
        return R_potential

    def choose_action(self, X, *args, **kwargs):
        if self.use_lvm:
            self.means = self.greedy(X)
            self.use_lvm = False
            self.arms = {}
            for i in range(self.n_arms):
                p = self.get_prior(i)
                self.arms[i] = {"mean": p["mean"][0], "var": p["var"][0]}
        return np.argmax(self.sample())


class RandomFourierFeatures:
    """Random Fourier Features class for approximating kernel functions.

    Args:
        n_features (int): Number of input features.
        n_components (int): Number of random Fourier features.
        gamma (float): Scaling factor for the random Fourier features.

    Attributes:
        W (ndarray): Randomly generated weights for the random Fourier features.
        b (ndarray): Randomly generated biases for the random Fourier features.
    """

    def __init__(self, n_features, n_components, gamma=1.0):
        self.n_features = n_features
        self.n_components = n_components
        self.gamma = gamma
        self.W = np.random.normal(0, 1 / np.sqrt(gamma), (n_features, n_components))
        self.b = np.random.uniform(0, 2 * np.pi, n_components)

    def transform(self, X):
        cos_part = np.cos(X.dot(self.W) + self.b)
        return np.sqrt(2 / self.n_components) * cos_part


class LinUCB_v1:
    r"""Linear Upper Confidence Bound policy

    :param int arm_num: number of arms
    :param int z_dim: dimension of Z (feature vector provided at each step)
    :param float delta: delta
    :param float lambda_reg: lambda for regularization
    :param Optional[str] name: alias name
    """

    def __init__(
        self,
        arm_num: int,
        z_dim: int,
        delta: float,
        lambda_reg: float,
        name: Optional[str] = None,
    ):
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta is expected within (0, 1). Got %.2f." % delta)
        if lambda_reg <= 0:
            raise ValueError(
                "lambda_reg is expected greater than 0. Got %.2f." % lambda_reg
            )

        self.arm_num = arm_num
        self.z_dim = z_dim
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.reset()

    def reset(self):
        # Each arm has its own Vt and theta_hat, both with dimensions based on z_dim
        self.Vt = [
            self.lambda_reg * np.eye(self.z_dim) for _ in range(self.arm_num)
        ]  # Covariance matrices per arm
        self.theta_hat = np.zeros(
            (self.arm_num, self.z_dim, 1)
        )  # Parameter vector for each arm, A x d x 1
        self.time = 1  # Current time step

    def __LinUCB(self, Z: np.ndarray) -> np.ndarray:
        """Compute the optimistic estimate of each arm's mean reward using LinUCB's upper confidence bound.

        :param Z: Context vector provided externally (d x 1)
        """
        # Confidence term based on the current time step
        root_beta_t = np.sqrt(self.lambda_reg) + np.sqrt(
            2 * np.log(1 / self.delta)
            + self.z_dim * np.log(1 + (self.time - 1) / (self.lambda_reg * self.z_dim))
        )

        # Compute the UCB for each arm
        ucb = np.zeros(self.arm_num)
        for arm in range(self.arm_num):
            theta_a = self.theta_hat[
                arm
            ].flatten()  # d-dimensional parameter vector for this arm
            V_a_inv = np.linalg.pinv(self.Vt[arm])  # Inverse of Vt for this arm
            ucb[arm] = theta_a.T @ Z + root_beta_t * np.sqrt(Z.T @ V_a_inv @ Z)

        return ucb

    def choose_action(self, Z: np.ndarray) -> int:
        """Choose an action based on the current upper confidence bound values given Z.

        :param Z: Z vector provided externally (d x 1)
        """
        ucb = self.__LinUCB(Z)
        return int(np.argmax(ucb))

    def update(self, chosen_action: int, Z: np.ndarray, reward: float, **kwargs):
        """Update model based on the chosen action, observed reward, and provided Z.

        :param pulled_arm_index: Index of the chosen arm
        :param Z: Z vector for the chosen action (d x 1)
        :param reward: Observed reward
        """
        pulled_arm_index = chosen_action
        # Update Vt and theta_hat for the chosen arm
        V_a = self.Vt[pulled_arm_index]
        V_a += Z @ Z.T
        self.Vt[pulled_arm_index] = V_a  # Update covariance matrix for the chosen arm

        # Update summation term for theta_hat for the chosen arm
        Xt = np.array([[reward]])  # Observed reward
        self.theta_hat[pulled_arm_index] = np.linalg.pinv(V_a) @ (
            self.theta_hat[pulled_arm_index] + Z * Xt
        )

        self.time += 1


class LinUCB_v2:
    r"""Linear Upper Confidence Bound policy

    :param int arm_num: number of arms
    :param int z_dim: dimension of Z (feature vector provided at each step)
    :param float delta: delta
    :param float lambda_reg: lambda for regularization
    :param Optional[str] name: alias name
    """

    def __init__(
        self,
        arm_num: int,
        z_dim: int,
        delta: float,
        lambda_reg: float,
        name: Optional[str] = None,
        **kwargs
    ):
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta is expected within (0, 1). Got %.2f." % delta)
        if lambda_reg <= 0:
            raise ValueError(
                "lambda_reg is expected greater than 0. Got %.2f." % lambda_reg
            )

        self.arm_num = arm_num
        self.z_dim = z_dim
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.reset()
        self.history = {}
        self.actions = []
        self.rewards = []

    def reset(self):
        self.Vt = [self.lambda_reg * np.eye(self.z_dim) for _ in range(self.arm_num)]
        self.theta_hat = np.zeros((self.arm_num, self.z_dim, 1))
        self.summation_AtXt = [np.zeros((self.z_dim, 1)) for _ in range(self.arm_num)]
        self.time = 1

    def __LinUCB(self, Z: np.ndarray) -> np.ndarray:
        """Compute the optimistic estimate of each arm's mean reward using LinUCB's upper confidence bound.

        :param Z: Context vector provided externally (d x 1)
        """
        # Confidence term based on the current time step
        root_beta_t = np.sqrt(self.lambda_reg) + np.sqrt(
            2 * np.log(1 / self.delta)
            + self.z_dim * np.log(1 + (self.time - 1) / (self.lambda_reg * self.z_dim))
        )

        # Compute the UCB for each arm
        ucb = np.zeros(self.arm_num)
        for arm in range(self.arm_num):
            theta_a = self.theta_hat[
                arm
            ].flatten()  # d-dimensional parameter vector for this arm
            V_a_inv = np.linalg.pinv(self.Vt[arm])  # Inverse of Vt for this arm
            ucb[arm] = theta_a.T @ Z + root_beta_t * np.sqrt(Z.T @ V_a_inv @ Z)

        return ucb

    def choose_action(self, Z: np.ndarray) -> int:
        """Choose an action based on the current upper confidence bound values given Z.

        :param Z: Z vector provided externally (d x 1)
        """
        if Z.shape[1] != 1:
            Z = Z[0].reshape(-1,1)
        ucb = self.__LinUCB(Z)
        return int(np.argmax(ucb))

    def update(self, chosen_action: int, Z: np.ndarray, reward: float, **kwargs):
        """
        Update the model with the observed reward for the chosen arm.

        :param pulled_arm_index: Index of the chosen arm
        :param Z: Context vector (z_dim x 1)
        :param reward: Observed reward for the chosen action
        """
        Z = Z.reshape(-1,1)
        pulled_arm_index = chosen_action
        V_a = self.Vt[pulled_arm_index]
        V_a += Z @ Z.T
        self.Vt[pulled_arm_index] = V_a

        self.summation_AtXt[pulled_arm_index] += Z * reward
        self.theta_hat[pulled_arm_index] = (
            np.linalg.pinv(V_a) @ self.summation_AtXt[pulled_arm_index]
        )
        self.time += 1
        self.actions.append(chosen_action)
        self.rewards.append(reward)


class SupLinUCB:
    """
    SupLinUCB: Supervised Linear Upper Confidence Bound with block-independent parameter updates

    :param int arm_num: number of arms
    :param int z_dim: dimension of Z (feature vector provided at each step)
    :param float delta: confidence parameter
    :param float lambda_reg: regularization parameter
    """

    def __init__(self, arm_num: int, z_dim: int, delta: float, lambda_reg: float):
        if not (0 < delta < 1):
            raise ValueError(f"Delta should be between 0 and 1. Got {delta}.")
        if lambda_reg <= 0:
            raise ValueError(f"lambda_reg should be greater than 0. Got {lambda_reg}.")

        self.arm_num = arm_num
        self.z_dim = z_dim
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.reset()

    def reset(self):
        self.V_matrices = [
            self.lambda_reg * np.eye(self.z_dim) for _ in range(self.arm_num)
        ]
        self.theta_vectors = [np.zeros((self.z_dim, 1)) for _ in range(self.arm_num)]
        self.summation_AtXt = [np.zeros((self.z_dim, 1)) for _ in range(self.arm_num)]
        self.time = 1
        self.block_time = {arm: 0 for arm in range(self.arm_num)}

    def _calculate_ucb(self, Z: np.ndarray) -> np.ndarray:
        """
        Calculate the Upper Confidence Bound for each arm based on the provided context Z.

        :param Z: Context vector (z_dim x 1)
        :return: UCB values for each arm
        """
        ucb_values = np.zeros(self.arm_num)
        for arm in range(self.arm_num):
            beta_t = np.sqrt(self.lambda_reg) + np.sqrt(
                2 * np.log(1 / self.delta)
                + self.z_dim
                * np.log(
                    1 + (self.block_time[arm] - 1) / (self.lambda_reg * self.z_dim)
                )
            )
            V_inv = np.linalg.pinv(self.V_matrices[arm])
            theta_a = self.theta_vectors[arm]
            ucb_values[arm] = theta_a.T @ Z + beta_t * np.sqrt((Z.T @ V_inv @ Z).item())

        return ucb_values

    def choose_action(self, Z: np.ndarray) -> int:
        """
        Choose an action based on the current Upper Confidence Bound values.

        :param Z: Context vector provided externally (z_dim x 1)
        :return: Index of the chosen arm
        """
        ucb_values = self._calculate_ucb(Z)
        return int(np.argmax(ucb_values))

    def update(self, chosen_action: int, Z: np.ndarray, reward: float, **kwargs):
        pulled_arm_index = chosen_action
        V_a = self.V_matrices[pulled_arm_index]
        V_a += Z @ Z.T
        self.V_matrices[pulled_arm_index] = V_a

        self.summation_AtXt[pulled_arm_index] += Z * reward
        self.theta_vectors[pulled_arm_index] = (
            np.linalg.pinv(V_a) @ self.summation_AtXt[pulled_arm_index]
        )
        self.block_time[pulled_arm_index] += 1


class RFB(LinUCB_v2):

    def __init__(
        self,
        n_arms,
        n_components,
        n_fourier,
        delta=0.1,
        lambda_reg=1.0,
        name=None,
        **kwargs,
    ):
        super().__init__(
            arm_num=n_arms,
            z_dim=n_fourier,
            delta=delta,
            lambda_reg=lambda_reg,
            name=name,
        )
        self.features = RandomFourierFeatures(
            n_features=n_components, n_components=n_fourier
        )
        self.reset()

    def choose_action(self, X):
        Z = self.features.transform(X[-1].flatten()).reshape(-1, 1)
        return super().choose_action(Z)

    def update(self, chosen_action, x_t, reward, **kwargs):
        Z = self.features.transform(x_t.flatten()).reshape(-1, 1)
        return super().update(chosen_action=chosen_action, Z=Z, reward=reward)


# Synthetic example comparing linear contextual bandits for stationary context
# (Non-contextual Thompson Sampling, Linear Thompson Sampling, and Warm-Started Linear Thompson Sampling)
def synthetic_contextual_comparison():
    def run_single_experiment_with_shared_context(
        X_log,
        A_log,
        R_log,
        X_t,
        true_thetas,
        d,
        k,
        T,
        R,
        lambda_,
        delta,
        alpha,
        num_samples,
    ):
        d = int(d)
        mab = MAB(n_arms=k, reward_sigma=R)

        cold_ts = LinearThompsonSampler(k, d, lambda_, delta, T, R)
        warm_ts = WarmStartLinearThompsonSampler(k, d, alpha, delta, T, R)
        warm_ts.initialize_prior_from_logs(X_log, A_log, R_log)

        regrets_cold, regrets_warm, regrets_mab = np.zeros(T), np.zeros(T), np.zeros(T)

        for t in range(T):
            rewards_true = [X_t @ true_thetas[i] for i in range(k)]

            arm_cold = cold_ts.choose_action(X_t)
            reward_cold = rewards_true[arm_cold] + np.random.normal(0, R)
            cold_ts.update(arm_cold, X_t, reward_cold)
            regrets_cold[t] = np.max(rewards_true) - rewards_true[arm_cold]

            arm_warm = warm_ts.choose_action(X_t)
            reward_warm = rewards_true[arm_warm] + np.random.normal(0, R)
            warm_ts.update(arm_warm, X_t, reward_warm)
            regrets_warm[t] = np.max(rewards_true) - rewards_true[arm_warm]

            arm_mab = mab.choose_action()
            reward_mab = rewards_true[arm_mab] + np.random.normal(0, R)
            mab.update(arm_mab, reward_mab)
            regrets_mab[t] = np.max(rewards_true) - rewards_true[arm_mab]

        return regrets_cold, regrets_warm, regrets_mab

    n_runs, T, d, k = 300, 1000, 5, 10
    lambda_, delta, alpha, R = 1.0, 0.1, 0.5, 0.25

    true_thetas = np.random.uniform(0.3, 0.8, size=(k, d))

    # Mean vector for shared context
    mu = np.full(d, 0.5)
    # Covariance matrix
    Sigma = (
        0.1 * np.eye(d)
        + 0.05 * np.triu(np.ones((d, d)), k=1)
        + 0.05 * np.tril(np.ones((d, d)), k=-1)
    )

    for warm_samples in [500]:
        X_log = np.random.multivariate_normal(mean=mu, cov=Sigma, size=warm_samples)
        A_log = np.random.randint(0, k, size=warm_samples)
        R_log = np.array(
            [
                X_log[i] @ true_thetas[A_log[i]] + np.random.normal(0, R)
                for i in range(warm_samples)
            ]
        )

        results_cold, results_warm, results_mab = [], [], []

        for _ in range(n_runs):
            X_t = np.random.multivariate_normal(mean=mu, cov=Sigma)
            rc, rw, rm = run_single_experiment_with_shared_context(
                X_log,
                A_log,
                R_log,
                X_t,
                true_thetas,
                d,
                k,
                T,
                R,
                lambda_,
                delta,
                alpha,
                warm_samples,
            )
            results_cold.append(rc)
            results_warm.append(rw)
            results_mab.append(rm)

        mean_regret_cold = np.mean(results_cold, axis=0)
        mean_regret_warm = np.mean(results_warm, axis=0)
        mean_regret_mab = np.mean(results_mab, axis=0)

        std_regret_cold = np.std(results_cold, axis=0)
        std_regret_warm = np.std(results_warm, axis=0)
        std_regret_mab = np.std(results_mab, axis=0)

        rounds = np.arange(T)

        import matplotlib.pyplot as plt

        plt.rc("font", size=24, family="serif")
        plt.style.use("tableau-colorblind10")
        plt.figure(figsize=(12, 5))
        plt.plot(rounds, mean_regret_cold, label="Linear Thompson Sampling")
        plt.fill_between(
            rounds,
            mean_regret_cold - std_regret_cold,
            mean_regret_cold + std_regret_cold,
            alpha=0.2,
        )

        plt.plot(
            rounds,
            mean_regret_warm,
            label="Linear Thompson Sampling (Warm-Start)",
            color="gray",
        )
        plt.fill_between(
            rounds,
            mean_regret_warm - std_regret_warm,
            mean_regret_warm + std_regret_warm,
            color="gray",
            alpha=0.2,
        )

        plt.plot(rounds, mean_regret_mab, label="Thompson Sampling")
        plt.fill_between(
            rounds,
            mean_regret_mab - std_regret_mab,
            mean_regret_mab + std_regret_mab,
            alpha=0.2,
        )

        plt.xlabel("Rounds")
        plt.ylabel("Expected Regret")
        plt.legend()
        plt.grid(alpha=0.5)
        plt.savefig("synthetic_contextual.pdf", bbox_inches="tight")
        plt.show()
