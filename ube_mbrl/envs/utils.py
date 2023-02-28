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

"""Tools to help define differentiable reward functions.
Taken from: https://github.com/sebascuri/rllib/blob/master/rllib/reward/utilities.py
Copyright (c) 2019 Sebastian Curi, licensed under MIT license,
cf. thirdparty_licenses.md file in the root directory of this source tree.
"""


import torch


def gaussian(x, value_at_1):
    """Apply an un-normalized Gaussian function with zero mean and scaled variance.
    Parameters
    ----------
    x : The points at which to evaluate_agent the Gaussian
    value_at_1: The reward magnitude when x=1. Needs to be 0 < value_at_1 < 1.
    """
    if type(value_at_1) is not torch.Tensor:
        value_at_1 = torch.tensor(value_at_1)
    scale = torch.sqrt(-2 * torch.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


def tolerance(x, lower, upper, margin=None):
    """Apply a tolerance function with optional smoothing.
    Can be used to design (smoothed) box-constrained reward functions.
    A tolerance function is returns 1 if x is in [lower, upper].
    If it is outside, it decays exponentially according to a margin.
    Parameters
    ----------
    x : the value at which to evaluate_agent the sparse reward.
    lower: The lower bound of the tolerance function.
    upper: The upper bound of the tolerance function.
    margin: A margin over which to smooth out the box-reward.
        If a positive margin is provided, uses a `gaussian` smoothing on the boundary.
    """
    if type(x) is not torch.Tensor:
        x = torch.tensor(x)
    if margin is None or margin == 0.0:
        in_bounds = (lower <= x) & (x <= upper)
        return in_bounds.type(torch.get_default_dtype())
    else:
        assert margin > 0
        diff = 0.5 * (upper - lower)
        mid = lower + diff

        # Distance is positive only outside the bounds
        distance = torch.abs(x - mid) - diff
        return gaussian(torch.relu(distance * (1 / margin)), value_at_1=0.1)
