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

import pathlib
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
import ube_mbrl.envs.reward_fns as ube_reward_fns
import ube_mbrl.envs.term_fns as ube_term_fns
import ube_mbrl.mbrl.env.termination_fns as term_fns
import ube_mbrl.mbrl.planning
import ube_mbrl.mbrl.types
import ube_mbrl.mbrl.util.common as mbrl_utils
from ube_mbrl.envs.truncate_wrapper import BulletTruncateWrapper
from ube_mbrl.mbrl.models import Ensemble, Model, ModelEnv, ModelTrainer, OneDTransitionRewardModel
from ube_mbrl.mbrl.planning import Agent
from ube_mbrl.mbrl.types import TransitionBatch
from ube_mbrl.mbrl.util import ReplayBuffer
from ube_mbrl.utils.video import VideoRecorder

PathType = Union[str, pathlib.Path]

ENV_NAME_TO_TERM_FN = {
    "InvertedPendulum-v2": "inverted_pendulum",
    "InvertedPendulumBulletEnv-v0": "inverted_pendulum",
    "InvertedPendulumSwingupBulletEnv-v0": "no_termination",
    "Pendulum-v1": "no_termination",
    "Hopper-v2": "hopper",
    "Hopper-v3": "hopper",
    "HalfCheetah-v2": "no_termination",
    "HalfCheetah-v3": "no_termination",
    "Walker2d-v2": "walker2d",
    "HalfCheetahBulletEnv-v0": "no_termination",
}


def fix_rng(
    env: gym.Env,
    seed: Optional[int] = 0,
    eval_env: Optional[gym.Env] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.random.Generator, torch.Generator]:
    """Fix the seed for all sources of randomness"""
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return rng, generator


def get_envs_from_name(env_name: str, params: dict) -> Tuple[str, gym.Env, gym.Env]:
    """
    Returns independent training and evaluation environments, along with the name of the env
    """
    if "TruncatedObs" in env_name:
        env_name = env_name.replace("TruncatedObs", "")
        env = BulletTruncateWrapper(gym.make(env_name))
        eval_env = BulletTruncateWrapper(gym.make(env_name))
    else:
        env = gym.make(env_name)
        eval_env = gym.make(env_name)

    # Config specific to SparsePendulum-v0
    if hasattr(env, "action_cost"):
        import ube_mbrl.envs.reward_fns

        env.set_action_cost(params["action_cost"])
        eval_env.set_action_cost(params["action_cost"])
        ube_mbrl.envs.reward_fns.ACTION_COST = params["action_cost"]

    if hasattr(env, "noise_std"):
        env.set_noise_std(params["noise_std"])
        eval_env.set_noise_std(params["noise_std"])

    return env_name, env, eval_env


def get_term_fn(env_name: str):
    if env_name == "SparsePendulum-v0":
        return ube_term_fns.sparse_pendulum_term_fn
    if env_name == "HopperBulletEnv-v0":
        return ube_term_fns.hopper_pybullet
    if env_name == "Walker2DBulletEnv-v0":
        return ube_term_fns.hopper_pybullet
    if env_name == "AntBulletEnv-v0":
        return ube_term_fns.ant_pybullet
    else:
        return getattr(term_fns, ENV_NAME_TO_TERM_FN[env_name], term_fns.no_termination)


def get_reward_fn(env_name: str):
    if env_name == "SparsePendulum-v0":
        return ube_reward_fns.sparse_pendulum
    else:
        return None


def set_device_in_hydra_cfg(device: str, cfg: dict):
    cfg["dynamics_model"]["device"] = str(device)
    cfg["dynamics_model"]["member_cfg"]["device"] = str(device)


def ensemble_to_envs(
    ensemble: Ensemble,
    env: gym.Env,
    rng: torch.Generator,
    reward_fn: ube_mbrl.mbrl.types.RewardFnType = None,
    termination_fn: Optional[ube_mbrl.mbrl.types.TermFnType] = term_fns.no_termination,
    add_mean_model: bool = False,
) -> List[ModelEnv]:
    """
    Take an ensemble of transition models and return a list of gym-like environments that we use to
    collect data. Internally, the gym environment will use the model dynamics to take steps, and use
    the reward function provided to calculate rewards.
    """
    # Unpack model config to replicate it for each member of the ensemble
    target_is_delta = ensemble.target_is_delta
    normalize = ensemble.input_normalizer is not None
    learned_rewards = ensemble.learned_rewards
    num_elites = ensemble.num_elites

    env_list = []
    for member in ensemble.model:
        member.device = ensemble.device
        wrapped_model = OneDTransitionRewardModel(
            member,
            target_is_delta=target_is_delta,
            normalize=normalize,
            learned_rewards=learned_rewards,
            num_elites=num_elites,
        )
        wrapped_model.input_normalizer = ensemble.input_normalizer
        env_list.append(ModelEnv(env, wrapped_model, termination_fn, reward_fn, rng))

    if add_mean_model:
        env_list.append(ModelEnv(env, ensemble, termination_fn, reward_fn, rng))
    return env_list


def add_ensemble_dim(batch_list: List[TransitionBatch]) -> TransitionBatch:
    """
    Takes a TransitionBatch list of size ensemble_size and returns a new TransitionBatch that
    stacks the list into a new numpy array dimension.
    """
    obs = np.stack([batch.obs for batch in batch_list], axis=0)
    act = np.stack([batch.act for batch in batch_list], axis=0)
    next_obs = np.stack([batch.next_obs for batch in batch_list], axis=0)
    rewards = np.stack([batch.rewards for batch in batch_list], axis=0)
    dones = np.stack([batch.dones for batch in batch_list], axis=0)
    return TransitionBatch(obs, act, next_obs, rewards, dones)


def train_model(
    model: Model,
    model_trainer: ModelTrainer,
    replay_buffer: ReplayBuffer,
    train_params: dict,
):
    """
    Train/eval dynamics model based on environment buffer.
    """
    dataset_train, dataset_val = mbrl_utils.get_basic_buffer_iterators(
        replay_buffer,
        batch_size=train_params["batch_size"],
        val_ratio=train_params["validation_ratio"],
        ensemble_size=len(model),
        shuffle_each_epoch=True,
        bootstrap_permutes=False,
    )
    model.update_normalizer(replay_buffer.get_all())
    model_trainer.train(
        dataset_train,
        dataset_val,
        # num_epochs=train_params["num_epochs"],
        patience=train_params["patience"],
    )


def step_env_and_add_to_buffer(
    env: gym.Env,
    obs: np.ndarray,
    agent: ube_mbrl.mbrl.planning.Agent,
    agent_kwargs: Dict,
    replay_buffer: ReplayBuffer,
) -> Tuple[np.ndarray, float, bool, Dict]:
    """
    Correctly handle the termination signal. If the termination comes from reaching a time limit,
    we end the episode but keep the done flag as False.
    """
    action = agent.act(obs, **agent_kwargs)
    next_obs, reward, done, info = env.step(action)
    time_truncated = info.get("TimeLimit.truncated", False)
    # We ignore the time termination signal for the Buffer transitions
    if time_truncated:
        done = False
    replay_buffer.add(obs, action, next_obs, reward, done)
    # From the outside, we terminate episodes if we reach the time limit or the done signal is True
    returned_done = time_truncated or done
    return next_obs, reward, returned_done, info


def evaluate(agent: Agent, env: gym.Env, dir: str, num_episodes: int, max_steps: int) -> dict:
    """
    Evaluate agent in a given RL environment for a number of episodes. In the first evaluation
    episode, the method saves a video of the agent's performance in the given directory.
    """
    video_recorder = VideoRecorder(dir)
    video_recorder.init(enabled=True)
    episode_returns = []
    for episode in range(num_episodes):
        obs = env.reset()
        agent.reset()
        video_recorder.init(enabled=(episode == 0))
        done = False
        episode_return = 0
        step = 0
        while not (done or step >= max_steps):
            action = agent.act(obs, sample=False)
            obs, reward, done, _ = env.step(action)
            video_recorder.record_default(env)
            episode_return += reward
            step += 1

        episode_returns.append(episode_return)
        video_recorder.save("agent.mp4")

    return {
        "avg_return": np.mean(episode_returns),
    }
