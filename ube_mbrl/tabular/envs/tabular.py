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

from gym import Env
from gym.spaces import Discrete
from gym.utils import seeding


class TabularEnv(Env):
    def __init__(self, num_states: int, num_actions: int):
        self.action_space = Discrete(num_actions)
        self.observation_space = Discrete(num_states)
        self.num_states = num_states
        self.num_actions = num_actions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
