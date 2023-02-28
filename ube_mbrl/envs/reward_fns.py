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

import torch
from .utils import tolerance

ACTION_COST = 0.0


def sparse_pendulum(act: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    assert len(obs.shape) == len(act.shape) == 2
    cos_th = obs[..., 0].unsqueeze(-1)
    th_dot = obs[..., 2].unsqueeze(-1)

    angle_tolerance = tolerance(cos_th, lower=0.95, upper=1.0, margin=0.1)
    velocity_tolerance = tolerance(th_dot, lower=-0.5, upper=0.5, margin=0.5)
    state_reward = angle_tolerance * velocity_tolerance

    action_tolerance = tolerance(act, lower=-0.1, upper=0.1, margin=0.1)
    action_cost = ACTION_COST * (action_tolerance - 1)

    cost = state_reward + action_cost
    return cost
