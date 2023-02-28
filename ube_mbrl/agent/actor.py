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
"""
Implementation modified from https://github.com/pranz24/pytorch-soft-actor-critic
Copyright (c) 2018 Pranjal Tandon, licensed under MIT license,
cf. thirdparty_licenses.md file in the root directory of this source tree.
"""

from typing import Tuple
import gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain=1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


class GaussianPolicy(nn.Module):
    """
    Control policy parameterized as a neural network whose output is a diagonal Gaussian
    distribution over actions, conditioned on the current state/observation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        params: dict,
        action_space: gym.Space = None,
    ):
        super(GaussianPolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.in_size = obs_dim
        self.num_layers = params["num_layers"]
        hid_size = params["hid_size"]
        activation_fn = params["activation_fn"]
        if activation_fn is None:
            activation_fn = nn.ReLU()
        else:
            activation_fn = hydra.utils.instantiate(activation_fn)

        # NN layers
        layers = [nn.Linear(self.in_size, hid_size), activation_fn]
        for _ in range(self.num_layers - 1):
            layers += [nn.Linear(hid_size, hid_size), activation_fn]
        self.hid_layers = nn.Sequential(*layers)

        self.mean_linear = nn.Linear(hid_size, action_dim)
        self.log_std_linear = nn.Linear(hid_size, action_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            high = np.array(action_space.high)
            low = np.array(action_space.low)
            self.action_scale = torch.FloatTensor((high - low) / 2.0)
            self.action_bias = torch.FloatTensor((high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hid_layers(obs)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_log_prob_from_batch(self, obs: torch.Tensor, act: torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        y_t = (act - self.action_bias) / self.action_scale
        # We clamp the value with some epsilon to avoid NaN in atanh
        y_t = torch.clamp(y_t, -1.0 + epsilon, 1.0 - epsilon)
        x_t = torch.atanh(y_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
