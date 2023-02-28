"""
Copyright (c) 2023 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from abc import ABC, abstractmethod


class Posterior(ABC):
    def __init__(self, prior_params: np.ndarray):
        self.params = prior_params

    @abstractmethod
    def update_params(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def get_mean(self):
        raise NotImplementedError


class DirichletPosterior(Posterior):
    """
    Class representing a Dirichlet distribution.
    Prior parameters correspond to independent dirichlet distributions.
    """

    def __init__(self, prior_params: np.ndarray):
        super().__init__(prior_params)

    def update_params(self, counts: np.ndarray):
        self.params += counts

    def sample(self, state: int, act: int):
        return np.random.dirichlet(self.params[state, act])

    def get_mean(self):
        return self.params / self.params.sum(axis=-1)[..., None]


class NormalGammaPosterior(Posterior):
    """
    Class representing a Normal-Gamma model of a random variable with unknown mean and variance.
    Prior parameters are specified for a multi-dimensional Normal-Gamma with independent dimensions.
    Each dimension has a set of parameters [mu, kappa, alpha, beta].
    The final format of the params is a np array of shape [input_dim1, input_dim2, 4]
    For RL, dim1 = states, dim2 = actions.
    """

    def __init__(self, prior_params: np.ndarray):
        super().__init__(prior_params)

    def update_params(self, counts, x_sum, xsquared_sums):
        n_sa = np.sum(counts, axis=-1)
        first_moment = x_sum / np.maximum(1, n_sa)
        second_moment = xsquared_sums / np.maximum(1, n_sa)

        mu, kappa, alpha, beta = np.rollaxis(self.params, axis=-1)
        mu_n = (kappa * mu + n_sa * first_moment) / (kappa + n_sa)
        kappa_n = kappa + n_sa
        alpha_n = alpha + 0.5 * n_sa
        var = second_moment - first_moment**2
        beta_n = beta + 0.5 * n_sa * (var + kappa * (first_moment - mu) ** 2 / (kappa + n_sa))
        self.params = np.stack((mu_n, kappa_n, alpha_n, beta_n), axis=-1)

    def sample(self, state: int, act: int):
        """
        We return the mean from the normal gamma
        """
        mu, kappa, alpha, beta = self.params[state, act]
        tau = np.random.gamma(shape=alpha, scale=beta ** (-1))
        mean = np.random.normal(loc=mu, scale=(kappa * tau) ** (-0.5))
        return mean

    def get_mean(self):
        """
        The mean is directly the mu parameter
        """
        return self.params[..., 0]


class NormalPosterior(Posterior):
    """
    Distribution specified by two parameters [mu, tau] representing the prior mean and precision of
    a normal distribution
    """

    def __init__(self, prior_params: np.ndarray, tau_x=1e6):
        """
        tau_x is the known precision of the random variable x. By default we assume a high value,
        corresponding to a random variable with very low variance.
        """
        self.tau_x = tau_x
        super().__init__(prior_params)

    def update_params(self, counts, x_sum, xsquared_sums):
        mu, tau = np.rollaxis(self.params, axis=-1)
        n_sa = np.sum(counts, axis=-1)
        tau_n = tau + n_sa * self.tau_x
        mu_n = (mu * tau + x_sum * self.tau_x) / tau_n
        self.params = np.stack((mu_n, tau_n), axis=-1)

    def sample(self, state: int, act: int):
        mu, tau = self.params[state, act]
        return mu + np.random.normal() * tau ** (-0.5)

    def get_mean(self):
        return self.params[..., 0]
