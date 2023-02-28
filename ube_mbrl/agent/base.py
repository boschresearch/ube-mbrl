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
from abc import ABC, abstractmethod
from typing import List, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ube_mbrl.agent.util as agent_util
import ube_mbrl.mbrl.models.util as models_util
from torch.optim import Adam
from ube_mbrl.agent.actor import GaussianPolicy
from ube_mbrl.agent.critic import QEnsemble
from ube_mbrl.mbrl.planning.core import Agent
from ube_mbrl.mbrl.types import TensorType, TransitionBatch
from ube_mbrl.mbrl.util.replay_buffer import ReplayBuffer

BufferType = Union[ReplayBuffer, List[ReplayBuffer]]


class BaseSAC(Agent, ABC):
    """
    Base class for SAC-like algorithms. It generalizes the double-critic architecture of SAC to
    support N critics instead. Supports both model-free and model-based agents through a common
    replay buffer interface. See `ube_mbrl.agent.SAC` for a concrete model-free implementation and
    `ube_mbrl.agent.QUSAC` for a concrete model-based implementation
    """

    AGENT_FNAME = "agent.pth"

    def __init__(self, env: gym.Env, device: torch.device, params: dict):
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.action_space = env.action_space
        self.device = device

        # All common parameters for SAC implementations
        self.batch_size = params["batch_size"]
        self.alpha = torch.tensor(params["alpha_temp"])
        self.tau = params["smoothness_coef"]
        self.auto_entropy_tuning = params["auto_entropy_tuning"]
        self.clip_grad_norm = params["clip_grad_norm"]
        self.target_entropy = -self.act_dim
        self.target_update_freq = params["target_update_freq"]
        self.actor_update_freq = params["actor_update_freq"]
        self.gamma = params["gamma"]
        self.critics_per_model = params["critics_per_model"]
        # Model-free SAC doesn't use dynamics ensemble, so ensemble size defaults to 1
        self.ensemble_size = params.get("ensemble_size", 1)

        # Common networks - Actor/Critic
        self.critic = QEnsemble(
            self.ensemble_size * self.critics_per_model,
            self.obs_dim,
            self.act_dim,
            params["critic"],
        ).to(self.device)
        self.critic_target = QEnsemble(
            self.ensemble_size * self.critics_per_model,
            self.obs_dim,
            self.act_dim,
            params["critic"],
        ).to(self.device)
        agent_util.hard_update(self.critic_target, self.critic)
        self.actor = GaussianPolicy(
            self.obs_dim, self.act_dim, params["actor"], self.action_space
        ).to(self.device)

        # Automatic entropy tuning
        if self.auto_entropy_tuning is True:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=3e-4)

        # To compute KL divergences and monitor policy changes
        self.old_actor = GaussianPolicy(
            self.obs_dim, self.act_dim, params["actor"], self.action_space
        ).to(self.device)
        agent_util.hard_update(self.old_actor, self.actor)

        # Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=params["actor"]["learning_rate"])
        self.critic_optim = Adam(self.critic.parameters(), lr=params["critic"]["learning_rate"])
        self.actor_updates = 1

    @abstractmethod
    def update_params(self, data: BufferType, step: int):
        raise NotImplementedError

    @abstractmethod
    def update_actor(self, batch: TransitionBatch, step: int):
        raise NotImplementedError

    def act(self, obs: np.ndarray, sample: bool = True, **_kwargs) -> np.ndarray:
        batched = obs.ndim > 1
        obs = torch.FloatTensor(obs)
        if not batched:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        # Return the mean action if not sampling from policy distribution
        if sample is True:
            action, _, _ = self.actor.sample(obs)
        else:
            _, _, action = self.actor.sample(obs)
        if batched:
            return action.detach().cpu().numpy()
        return action.detach().cpu().numpy()[0]

    def update_critic(self, ensemble_batch: TransitionBatch, step: int) -> torch.Tensor:
        obs, act, next_obs, reward, done = ensemble_batch.astuple()

        state = models_util.to_tensor(obs).to(self.device)
        next_state = models_util.to_tensor(next_obs).to(self.device)
        act = models_util.to_tensor(act).to(self.device)
        reward = models_util.to_tensor(reward).to(self.device)
        done = models_util.to_tensor(done).to(self.device)

        # Add dimension to reward and done_mask
        reward = reward.unsqueeze(dim=-1)
        done_mask = (~done).unsqueeze(dim=-1)

        # Targets for Q-function loss
        with torch.no_grad():
            next_act, next_act_log_pi, _ = self.actor.sample(next_state)
            q_next_values = self.get_min_q(next_state, next_act, self.critic_target)
            qf_next_target = q_next_values - self.alpha * next_act_log_pi
            target_q_values = reward + done_mask * self.gamma * qf_next_target

        q_values = self.critic(self.get_critic_input(self.interleave(state), self.interleave(act)))
        loss = (
            F.mse_loss(q_values, self.interleave(target_q_values), reduction="none")
            .sum((1, 2))
            .sum()
        )
        self.critic_optim.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(list(self.critic.parameters()), max_norm=0.5)
        self.critic_optim.step()
        return loss.item()

    def maybe_update_entropy_gain(self, log_pi):
        """
        Update parameter alpha that controls gain of entropy loss
        """
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

    def get_min_q(
        self, state: torch.Tensor, action: torch.Tensor, critic: QEnsemble
    ) -> torch.Tensor:
        """
        Two scenarios:

        (1) Model-free SAC: state and action have dimensions (batch_size x input_dim). We add a
            dummy ensemble dimension of size 1 and return the minimum among the `critics_per_model`
            value predictions.

        (2) Model-based SAC: state and action have dimensions (ensemble x batch_size x input_dim).
            For each ensemble dimension (corresponding to a specific dynamics model) we get the
            `critics_per_model` value predictions and compute the minimum.
        """
        assert state.ndim == action.ndim
        critic_ensemble_size = critic.num_members // self.critics_per_model
        if state.ndim == 2:
            state = state.repeat(critic_ensemble_size, 1, 1)
            action = action.repeat(critic_ensemble_size, 1, 1)
        q_values = critic(self.get_critic_input(self.interleave(state), self.interleave(action)))
        return self.compute_min(q_values, critic_ensemble_size)

    def compute_min(self, tensor: torch.Tensor, ensemble_size: int) -> torch.Tensor:
        """
        Reduces the input tensor (coming from Q values or Q targets) with the min operation over the
        critics per model, i.e., If we have 5 dynamic models and 2 critics per model, we compute the
        min value of the 2 critics of each model
        """
        batch_size = tensor.size(dim=1)
        return torch.min(
            torch.reshape(tensor, (ensemble_size, self.critics_per_model, batch_size, 1)),
            dim=1,
        )[0]

    def interleave(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor (ensemble x batch x input_dim) it repeats the tensor with interleave along
        the ensemble dimension to obtain a new tensor (ensemble * critics_per_model x batch x
        input_dim). For input tensors of size (batch x input_dim) this method returns a new tensor
        of size (critics_per_model x batch x input_dim), and the interleaving is equivalent as
        vanilla repeat.
        """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(dim=0)
        return tensor.repeat_interleave(self.critics_per_model, dim=0)

    def get_critic_input(self, obs: TensorType, act: TensorType) -> torch.Tensor:
        return torch.cat([obs, act], dim=obs.ndim - 1).float()

    def save(self, dir: str):
        torch.save(self, pathlib.Path(dir) / self.AGENT_FNAME)
