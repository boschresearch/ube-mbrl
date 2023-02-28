"""
Copyright (c) 2023 Robert Bosch GmbH
This source code is derived from mbrl-lib
    (https://github.com/facebookresearch/mbrl-lib/tree/4543fc929321fdd6e6522528c68e54d822ad2a6a)
Copyright (c) Facebook, Inc. and its affiliates, licensed under the MIT license,
cf. thirdparty_licenses.md file in the root directory of this source tree.
"""

import abc
import pathlib
from typing import Any, Union

import gym
import hydra
import numpy as np
import omegaconf

import ube_mbrl.mbrl.models
import ube_mbrl.mbrl.types


class Agent:
    """Abstract class for all agents."""

    @abc.abstractmethod
    def act(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Args:
            obs (np.ndarray): the observation for which the action is needed.

        Returns:
            (np.ndarray): the action.
        """
        pass

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Unless overridden by a child class, this will be equivalent to :meth:`act`.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """

        return self.act(obs, **_kwargs)

    def reset(self):
        """Resets any internal state of the agent."""
        pass


class RandomAgent(Agent):
    """An agent that samples action from the environments action space.

    Args:
        env (gym.Env): the environment on which the agent will act.
    """

    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, *_args, **_kwargs) -> np.ndarray:
        """Issues an action given an observation.

        Returns:
            (np.ndarray): an action sampled from the environment's action space.
        """
        return self.env.action_space.sample()


def complete_agent_cfg(
    env: Union[gym.Env, ube_mbrl.mbrl.models.ModelEnv], agent_cfg: omegaconf.DictConfig
):
    """Completes an agent's configuration given information from the environment.

    The goal of this function is to completed information about state and action shapes and ranges,
    without requiring the user to manually enter this into the Omegaconf configuration object.

    It will check for and complete any of the following keys:

        - "obs_dim": set to env.observation_space.shape
        - "action_dim": set to env.action_space.shape
        - "action_range": set to max(env.action_space.high) - min(env.action_space.low)
        - "action_lb": set to env.action_space.low
        - "action_ub": set to env.action_space.high

    Note:
        If the user provides any of these values in the Omegaconf configuration object, these
        *will not* be overridden by this function.

    """
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    def _check_and_replace(key: str, value: Any, cfg: omegaconf.DictConfig):
        if key in cfg.keys() and key not in cfg:
            setattr(cfg, key, value)

    _check_and_replace("num_inputs", obs_shape[0], agent_cfg)
    if "action_space" in agent_cfg.keys() and isinstance(
        agent_cfg.action_space, omegaconf.DictConfig
    ):
        _check_and_replace("low", env.action_space.low.tolist(), agent_cfg.action_space)
        _check_and_replace("high", env.action_space.high.tolist(), agent_cfg.action_space)
        _check_and_replace("shape", env.action_space.shape, agent_cfg.action_space)

    if "obs_dim" in agent_cfg.keys() and "obs_dim" not in agent_cfg:
        agent_cfg.obs_dim = obs_shape[0]
    if "action_dim" in agent_cfg.keys() and "action_dim" not in agent_cfg:
        agent_cfg.action_dim = act_shape[0]
    if "action_range" in agent_cfg.keys() and "action_range" not in agent_cfg:
        agent_cfg.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]
    if "action_lb" in agent_cfg.keys() and "action_lb" not in agent_cfg:
        agent_cfg.action_lb = env.action_space.low.tolist()
    if "action_ub" in agent_cfg.keys() and "action_ub" not in agent_cfg:
        agent_cfg.action_ub = env.action_space.high.tolist()

    return agent_cfg
