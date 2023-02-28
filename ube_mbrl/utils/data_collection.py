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

from typing import Dict, List, Optional, Union, cast

import numpy as np
import torch
from ube_mbrl.mbrl.models import BasicEnsemble, ModelEnv
from ube_mbrl.mbrl.planning import Agent
from ube_mbrl.mbrl.types import TransitionBatch
from ube_mbrl.mbrl.util import ReplayBuffer

ModelBufferType = Union[ReplayBuffer, List[ReplayBuffer]]


def collect_ensemble_model_transitions(
    ensemble_envs: List[ModelEnv],
    agent: Agent,
    replay_buffer: List[ReplayBuffer],
    env_replay_buffer: ReplayBuffer,
    rollout_length: int,
    batch_size: int,
    use_mean_model_buffer: Optional[bool] = False,
    rollout_mode: str = "random_model",
):
    # We temporarily set the desired rollout mode and then reset to the original Traverse the list
    # of buffers in reverse order (so that the mean_model is first) in case we want to use the
    # mean_model_buffer as initial state distribution.
    for i, (env, buffer) in enumerate(zip(ensemble_envs[::-1], replay_buffer[::-1])):
        if i > 0 and use_mean_model_buffer:
            init_state_dist = replay_buffer[-1]
        else:
            init_state_dist = env_replay_buffer
        if isinstance(env.dynamics_model.model, BasicEnsemble):
            env.dynamics_model.set_propagation_method(rollout_mode)
        collect_single_model_transitions(
            env, agent, buffer, rollout_length, batch_size, init_state_dist=init_state_dist
        )
        env.dynamics_model.set_propagation_method()


def collect_single_model_transitions(
    model_env: ModelEnv,
    agent: Agent,
    replay_buffer: ReplayBuffer,
    rollout_horizon: int,
    batch_size: int,
    agent_kwargs: Dict = {},
    init_state_dist: Optional[ReplayBuffer] = None,
):
    """
    Collects data from rollouts under a learned transition model in a replay buffer.
    """
    if init_state_dist is not None:
        batch = init_state_dist.sample(batch_size)
        initial_obs, *_ = cast(TransitionBatch, batch).astuple()
    else:
        initial_obs = np.array([model_env.observation_space.sample() for _ in range(batch_size)])

    model_state = model_env.reset(
        initial_obs_batch=cast(np.ndarray, initial_obs),
        return_as_np=True,
    )
    accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
    obs = initial_obs
    for _ in range(rollout_horizon):
        action = agent.act(obs, **agent_kwargs)
        pred_next_obs, pred_rewards, pred_dones, model_state = model_env.step(
            action, model_state, sample=True
        )
        replay_buffer.add_batch(
            obs[~accum_dones],
            action[~accum_dones],
            pred_next_obs[~accum_dones],
            pred_rewards[~accum_dones, 0],
            pred_dones[~accum_dones, 0],
        )
        obs = pred_next_obs
        accum_dones |= pred_dones.squeeze()
