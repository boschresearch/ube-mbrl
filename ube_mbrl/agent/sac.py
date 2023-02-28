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

from typing import cast

import gym
import torch
import torch.nn as nn
import ube_mbrl.agent.util as agent_util
import ube_mbrl.mbrl.models.util as models_util
from ube_mbrl.agent.base import BaseSAC, BufferType
from ube_mbrl.mbrl.types import TransitionBatch


class SAC(BaseSAC):
    """
    A standard model-free SAC agent with support for N critics.
    """

    def __init__(self, env: gym.Env, device: torch.device, params: dict):
        super().__init__(env, device, params)

    def update_params(self, data: BufferType, step: int):
        batch = cast(TransitionBatch, data.sample(self.batch_size))
        self.update_critic(batch, step)
        if step % self.actor_update_freq == 0:
            self.update_actor(batch, step)
            self.actor_updates += 1
        if step % self.target_update_freq == 0:
            agent_util.soft_update(self.critic_target, self.critic, self.tau)

    def update_actor(self, batch: TransitionBatch, step: int):
        # Get actions/log_probs from batch
        obs, *_ = batch.astuple()
        state = models_util.to_tensor(obs).to(self.device)
        pi, log_pi, _ = self.actor.sample(state)

        # Compute min of Q-values
        min_q_values = self.get_min_q(state, pi, self.critic)

        # Actor loss: maximize min of Q-values and entropy of policy
        actor_loss = ((self.alpha * log_pi) - (min_q_values)).mean()

        # Reset gradients, do backward pass, clip gradients (if enabled) and take a gradient step
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(list(self.actor.parameters()), max_norm=0.5)
        self.actor_optim.step()

        # If we are using auto-entropy tuning, then we update the alpha gain
        self.maybe_update_entropy_gain(log_pi)
        return actor_loss.item()
