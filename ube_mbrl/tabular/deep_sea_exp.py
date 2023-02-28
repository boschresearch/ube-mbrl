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

import matplotlib.pyplot as plt
import numpy as np
from ube_mbrl.mbrl.util import ReplayBuffer
from ube_mbrl.tabular.config import default_parameters

import util
from agent import TabularAgent
from envs.deep_sea import DeepSea


def train(parameters):
    length = parameters["deep_sea_size"]
    env = DeepSea(length=length)
    obs_shape = (1,)
    act_shape = (1,)

    # Fix RNG
    seed = parameters["seed"]
    rng = util.fix_rng(env, seed=seed)

    agent = TabularAgent(env, params=parameters["agent"])

    num_episodes = parameters["num_episodes"]
    ep_returns = []
    total_regret = np.zeros(num_episodes)
    solved_count = 0
    solved_episode = 0

    # We define an episodic return over which the task has been solved
    solved_return_treshold = 0.5

    # Additionally, we declare the task solved when the percentage of episodes solved is above 10%
    solved_percentage = 0.1
    for i in range(num_episodes):
        ep_buffer = ReplayBuffer(
            length, obs_shape, act_shape, obs_type=np.int32, action_type=np.int32, rng=rng
        )
        obs = env.reset()
        ep_return = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            ep_buffer.add(obs, action, next_obs, reward, done)
            ep_return += reward
            obs = next_obs

        if ep_return < solved_return_treshold:
            total_regret[i] = total_regret[i - 1] + 1
        else:
            total_regret[i] = total_regret[i - 1]
            solved_count += 1

        print(f"Episode #{i}: ep_return = {ep_return}")
        ep_returns.append(ep_return)

        if solved_count / (i + 1) > solved_percentage and solved_episode == 0:
            solved_episode = i

        # Update agent with new data
        agent.update(ep_buffer, t=i + 1)

    # Plot total regret
    fix, ax = plt.subplots()
    ax.plot(total_regret)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total regret")
    plt.show()


if __name__ == "__main__":
    train(default_parameters)
