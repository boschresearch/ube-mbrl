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


def sparse_pendulum_term_fn(act: torch.Tensor, next_obs: torch.Tensor):
    assert len(next_obs.shape) == 2
    velocity = next_obs[:, 2]
    done = torch.any(torch.abs(velocity) > 200, dim=-1) | torch.any(torch.abs(act) > 200, dim=-1)
    done = done[:, None]
    return done


def mountain_car_continuous(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    position = next_obs[:, 0]
    velocity = next_obs[:, 1]
    done = (position >= 0.45) * (velocity >= 0.0)
    done = done[:, None]
    return done


def mountain_car_continuous(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2


def hopper_pybullet(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    initial_height = 1.25
    pitch = next_obs[:, 7]
    not_done = (
        torch.isfinite(next_obs).all(-1) * (height + initial_height > 0.8) * (pitch.abs() < 1.0)
    )

    done = ~not_done
    done = done[:, None]
    return done


def ant_pybullet(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    height = next_obs[:, 0]
    initial_height = 0.75
    not_done = torch.isfinite(next_obs).all(-1) * (height + initial_height > 0.26)

    done = ~not_done
    done = done[:, None]
    return done
