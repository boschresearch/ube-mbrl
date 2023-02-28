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

from gym import spaces
from gym.core import ObservationWrapper
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv


class BulletTruncateWrapper(ObservationWrapper):
    def __init__(self, env: WalkerBaseBulletEnv):
        super().__init__(env)
        self.num_contacts = len(self.robot.foot_list)
        self.observation_space = spaces.Box(
            low=env.observation_space.low[: -self.num_contacts],
            high=env.observation_space.high[: -self.num_contacts],
        )

    def observation(self, observation):
        return observation[: -self.num_contacts]
