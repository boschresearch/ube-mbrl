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

import os

import pybullet_envs
import torch
import ube_mbrl.envs
import ube_mbrl.mbrl.util.common as mbrl_utils
import ube_mbrl.utils.common as utils_common
from ube_mbrl.agent import SAC
from ube_mbrl.conf import sac_online_default_params
from ube_mbrl.mbrl.util import ReplayBuffer
from ube_mbrl.utils.video import close_virtual_display


def train(parameters):
    _, env, eval_env = utils_common.get_envs_from_name(parameters["env_name"], parameters)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Fix RNG
    seed = parameters["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng, generator = utils_common.fix_rng(env, eval_env=eval_env, seed=seed, device=device)

    # Create replay buffer to store environment interactions
    env_buffer_capacity = int(parameters["env_buffer_capacity"])
    env_replay_buffer = ReplayBuffer(env_buffer_capacity, obs_shape, act_shape, rng=rng)

    # Create SAC agent
    device = torch.device(device)
    agent = SAC(env, device, parameters["agent"])

    # Pre-fill the replay buffer with some initial data
    buffer_init_steps = parameters["buffer_init_steps"]
    mbrl_utils.rollout_agent_trajectories(
        env, buffer_init_steps, agent, {}, replay_buffer=env_replay_buffer
    )

    # Directory for creating agent checkpoints
    artifact_dir = os.getcwd()
    checkpoints_dir = os.path.join(artifact_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Agent training loop
    num_steps = int(parameters["num_steps"])
    steps_per_epoch = int(parameters["steps_per_epoch"])
    global_step = 0
    agent_updates = 1
    while global_step <= num_steps:
        obs, done = env.reset(), False
        ep_step = 0
        while not (done or ep_step >= steps_per_epoch):
            next_obs, _, done, _ = utils_common.step_env_and_add_to_buffer(
                env, obs, agent, {}, env_replay_buffer
            )
            for _ in range(parameters["agent_updates_per_step"]):
                agent.update_params(env_replay_buffer, agent_updates)
                agent_updates += 1

            # Evaluate agent
            if (global_step + 1) % parameters["freq_agent_eval"] == 0:
                epoch_dir = os.path.join(checkpoints_dir, f"step_{global_step+1}")
                os.makedirs(epoch_dir, exist_ok=True)
                eval_dict = utils_common.evaluate(
                    agent,
                    eval_env,
                    epoch_dir,
                    num_episodes=parameters["eval_episodes"],
                    max_steps=steps_per_epoch,
                )
                ep = global_step // steps_per_epoch
                print(f"Episode {ep} return = {eval_dict['avg_return']}")
                agent.save(epoch_dir)

            obs = next_obs
            ep_step += 1
            global_step += 1
            if global_step > num_steps:
                break

    # Close env and pyvirtual display before exiting
    env.close()
    eval_env.close()
    close_virtual_display()


if __name__ == "__main__":
    train(sac_online_default_params)
