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
from ube_mbrl.mbrl.models.util import EnsembleLinearLayer

# Initialize Critic weights
def weights_init_(m):
    if isinstance(m, EnsembleLinearLayer):
        # We use a larger initialization gain to promote diversity in the ensemble of critics.
        nn.init.orthogonal_(m.weight, gain=10.0)
        nn.init.constant_(m.bias, 0)


class QEnsemble(nn.Module):
    """
    The critic is parameterized as an ensemble of neural networks.

    Model-free RL: generalizes the notion of twin networks in SAC to arbitrary N

    Model-based RL: allows training individual members of the Q-ensemble with different replay
    buffers, each coming from a specific dynamics model.
    """

    def __init__(self, ensemble_size: int, obs_dim: int, action_dim: int, params: dict):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.in_size = obs_dim + action_dim
        self.out_size = 1
        self.num_members = ensemble_size
        self.num_layers = params["num_layers"]
        hid_size = params["hid_size"]
        activation_fn = params["activation_fn"]
        if activation_fn is None:
            activation_fn = nn.ReLU()
        else:
            activation_fn = hydra.utils.instantiate(activation_fn)

        def create_ensemble_linear_layer(in_size, out_size):
            return EnsembleLinearLayer(self.num_members, in_size, out_size)

        # First layer
        hidden_layers = [
            nn.Sequential(create_ensemble_linear_layer(self.in_size, hid_size)),
            activation_fn,
        ]

        # Hidden layers
        for _ in range(self.num_layers - 1):
            hidden_layers.append(
                nn.Sequential(create_ensemble_linear_layer(hid_size, hid_size), activation_fn)
            )

        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Output layer
        self.out = create_ensemble_linear_layer(hid_size, self.out_size)

        self.apply(weights_init_)

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layers(obs_action)
        return self.out(x)

    def mean_forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        """
        Pass the same [obs, action] batch through all members of the ensemble and aggregate the
        ensemble prediction of the values by returning the mean.
        """
        obs_action = obs_action.unsqueeze(dim=0).repeat(self.num_members, 1, 1)
        q_values = self.forward(obs_action)
        return torch.mean(q_values, dim=0)
