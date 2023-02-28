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

import random
from typing import Optional

import gym
import numpy as np


def fix_rng(
    env: gym.Env,
    seed: Optional[int] = 0,
):
    env.seed(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed=seed)
    random.seed(seed)
    return rng


def sample_mean_from_normal_gamma(mu, kappa, alpha, beta):
    tau = np.random.gamma(shape=alpha, scale=beta ** (-1))
    mean = np.random.normal(loc=mu, scale=(kappa * tau) ** (-0.5))
    return mean
