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

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Uncertainty Net weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


MIN_VAR = 1e-1


class QUncertainty(nn.Module):
    """
    The critic uncertainty network approximates the solution to the uncertainty Belman equation.

    Given a batch of observations and actions, it outputs the predicted variance of Q-values,
    associated with the epistemic uncertainty due to limited data.
    """

    def __init__(self, size: int, obs_dim: int, action_dim: int, params: dict):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.in_size = obs_dim + action_dim
        self.out_size = 1
        self.num_members = size
        self.num_layers = params["num_layers"]
        hid_size = params["hid_size"]
        activation_fn = params["activation_fn"]
        self.output_fn = params["output_fn"]
        if activation_fn is None:
            activation_fn = nn.ReLU()
        else:
            activation_fn = hydra.utils.instantiate(activation_fn)

        # NN layers
        layers = [nn.Linear(self.in_size, hid_size), activation_fn]
        for _ in range(self.num_layers - 1):
            layers += [nn.Linear(hid_size, hid_size), activation_fn]
        layers.append(nn.Linear(hid_size, self.out_size))

        self.variance = nn.Sequential(*layers)
        self.min_var = nn.Parameter(MIN_VAR * torch.ones(1, self.out_size), requires_grad=False)
        self.apply(weights_init_)

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        self.out = self.variance(obs_action)
        if self.output_fn == "huber_loss":
            return (
                F.huber_loss(self.out, torch.zeros_like(self.out), reduction="none") + self.min_var
            )
        elif self.output_fn == "smooth_abs":
            epsilon = 1e-3
            return torch.sqrt(torch.pow(self.out, 2) + epsilon) + self.min_var
        elif self.output_fn == "softplus":
            return F.softplus(self.out) + self.min_var
