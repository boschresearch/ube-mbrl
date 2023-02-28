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

from typing import Tuple, cast

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import ube_mbrl.agent.util as agent_util
import ube_mbrl.mbrl.models.util as models_util
import ube_mbrl.mbrl.types
import ube_mbrl.mbrl.util.math
import ube_mbrl.utils.common as utils_common
from torch.optim import Adam
from ube_mbrl.agent.base import BaseSAC, BufferType
from ube_mbrl.agent.uncertainty import MIN_VAR, QUncertainty
from ube_mbrl.mbrl.models import Ensemble
from ube_mbrl.mbrl.types import TransitionBatch


class QUSAC(BaseSAC):
    """
    Q-uncertainty SAC. A model-based version of SAC with optional uncertainty quantification. Uses
    an ensemble of critics, each trained with independent dynamic models, to estimate optimistic
    Q-values which are then maximized by the actor.
    """

    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        params: dict,
        dynamics_model: Ensemble,
        reward_fn: ube_mbrl.mbrl.types.RewardFnType = None,
    ):
        """
        The reward fn is optional. In case the rewards are assumed to be known, we use the reward
        function during uncertainty reward estimation.
        """
        super().__init__(env, device, params)
        self.dynamics_model = dynamics_model
        self.reward_fn = reward_fn
        self.uncertainty_type = params["uncertainty_type"]
        self.use_ube_target = params["use_ube_target"]
        self.uncertainty_penalty = params["uncertainty_penalty"]
        self.act_n_samples = params["ube"]["act_n_samples"]
        self.ube_regularization = params["ube"]["regularization_penalty"]

        if self.uncertainty_type not in ("none", "ensemble"):
            self.uncertainty = QUncertainty(
                self.ensemble_size, self.obs_dim, self.act_dim, params["ube"]
            ).to(self.device)
            if self.use_ube_target:
                self.uncertainty_target = QUncertainty(
                    self.ensemble_size, self.obs_dim, self.act_dim, params["ube"]
                ).to(self.device)
                agent_util.hard_update(self.uncertainty_target, self.uncertainty)
            self.uncertainty_optim = Adam(
                self.uncertainty.parameters(), lr=params["ube"]["learning_rate"]
            )

    def update_params(self, data: BufferType, step: int):
        # Sample batches from the model buffers
        batch_list = [buffer.sample(self.batch_size) for buffer in data]
        batch = [cast(TransitionBatch, batch) for batch in batch_list]

        # Split the list of transition batch into the inidividual model batches (used for training
        # the critic), and the mean model batch, used to train the UBE and actor
        model_batch = batch[: len(self.dynamics_model)]
        model_batch = utils_common.add_ensemble_dim(model_batch)
        mean_model_batch = batch[-1]

        # Critic / UBE updates
        if self.uncertainty_type not in ("none", "ensemble"):
            self.update_ube(mean_model_batch, step)
        self.update_critic(model_batch, step)

        # Target updates
        if step % self.target_update_freq == 0:
            agent_util.soft_update(self.critic_target, self.critic, self.tau)
            if self.uncertainty_type not in ("none", "ensemble") and self.use_ube_target:
                agent_util.soft_update(self.uncertainty_target, self.uncertainty, self.tau)

        # Actor update
        if step % self.actor_update_freq == 0:
            self.update_actor(mean_model_batch, step)
            self.actor_updates += 1

    def update_actor(self, batch: TransitionBatch, step: int):
        # Get actions/log_probs from batch
        obs, *_ = batch.astuple()
        state = models_util.to_tensor(obs).to(self.device)
        pi, log_pi, _ = self.actor.sample(state)

        # Compute Q-values + uncertainty
        q_values = self.get_min_q(state, pi, self.critic)
        mean_q_values = torch.mean(q_values, dim=0)
        if self.uncertainty_type not in ("none", "ensemble"):
            std_q_values = torch.sqrt(self.uncertainty(self.get_critic_input(state, pi)))
        elif self.uncertainty_type == "ensemble":
            q_values = self.get_min_q(state, pi, self.critic) if q_values is None else q_values
            std_q_values = torch.std(q_values, dim=0)
        else:
            std_q_values = torch.zeros_like(mean_q_values)
        q_uncertain_values = mean_q_values + self.uncertainty_penalty * std_q_values

        # Actor loss: maximize optimistic values and entropy of policy
        actor_loss = (self.alpha * log_pi - q_uncertain_values).mean()

        # Reset gradients, do backward pass, clip gradients (if enabled) and take a gradient step
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(list(self.actor.parameters()), max_norm=0.5)
        self.actor_optim.step()

        # If we are using auto-entropy tuning, then we update the alpha gain
        self.maybe_update_entropy_gain(log_pi)
        return actor_loss.item()

    def update_ube(self, mean_model_batch: TransitionBatch, step: int) -> torch.Tensor:
        obs, act, next_obs, reward, done = mean_model_batch.astuple()

        state = models_util.to_tensor(obs).to(self.device)
        next_state = models_util.to_tensor(next_obs).to(self.device)
        act = models_util.to_tensor(act).to(self.device)
        reward = models_util.to_tensor(reward).to(self.device)
        done = models_util.to_tensor(done).to(self.device)

        done_mask = (~done).unsqueeze(dim=-1)
        gamma = self.gamma**2

        # Compute target
        with torch.no_grad():
            next_act, *_ = self.actor.sample(next_state)
            next_act = next_act.to(self.device)
            var_reward, u = self.compute_uncertainty_rewards(state, act)
            if self.use_ube_target:
                next_ube_out = self.uncertainty_target(self.get_critic_input(next_state, next_act))
            else:
                next_ube_out = self.uncertainty(self.get_critic_input(next_state, next_act))
            target = var_reward + gamma * (u + done_mask * next_ube_out)

        u_values = self.uncertainty(self.get_critic_input(state, act))
        # Regularize network to penalize negative values before softplus
        pre_soft_plus_out = self.uncertainty.out
        min_var_threshold = MIN_VAR * torch.ones_like(pre_soft_plus_out)
        negative_features_loss = torch.pow(F.relu(-(pre_soft_plus_out - min_var_threshold)), 2)

        loss = (
            F.mse_loss(u_values, target, reduction="none").sum()
            + self.ube_regularization * negative_features_loss.sum()
        )
        self.uncertainty_optim.zero_grad()
        loss.backward()
        if self.clip_grad_norm:
            nn.utils.clip_grad_norm_(list(self.uncertainty.parameters()), max_norm=0.5)
        self.uncertainty_optim.step()
        return loss.item()

    def compute_uncertainty_rewards(
        self, state: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.uncertainty_type == "pombu":
            return self.pombu_rewards(state, act)
        if self.uncertainty_type in ["exact_ube_2", "exact_ube_3"]:
            return self.exact_ube_rewards(state, act)
        else:
            raise ValueError(f"Invalid reward type {self.uncertainty_type} for UBE")

    def pombu_rewards(
        self, state: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the local uncertainty measure defined in the POMBU paper.
        See: https://arxiv.org/abs/1911.12574

        Args:
            state (torch.Tensor): batch of states of size (batch_size x state_size)
            action (torch.Tensor): batch of actions that were taken during data collection, i.e.,
            this comes directly from the training data, size is (batch_size x act_size)

        Returns:
            torch.Tensor: absolute local uncertainty for each input state, size is (batch_size x 1)
        """
        var_reward, next_state = self.ensemble_prediction(state, act)
        return var_reward, self.var_mean_value(next_state)

    def var_mean_value(self, next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute the variance over next_state predictions of the mean value of the critic ensemble.
        This requires passing the next_state predictions through each member of the critic ensemble,
        compute its mean, and then compute the variance over the dynamics model ensemble dimension.

        Args:
            next_state (torch.Tensor): forward predictions for the next state (ensemble x batch_size
            x state_size)

        Returns:
            torch.Tensor: returns epistemic variance (over the dynamics ensemble) of the critic mean
            prediction
        """
        next_state_reshaped = torch.reshape(
            next_state, (self.ensemble_size * self.batch_size, self.obs_dim)
        )
        next_act, _, _ = self.actor.sample(next_state_reshaped)
        mean_value = self.get_mean_value(next_state_reshaped, next_act)
        mean_value = torch.reshape(mean_value, (self.ensemble_size, self.batch_size, 1))
        return torch.var(mean_value, dim=0)

    def exact_ube_rewards(
        self, state: torch.Tensor, act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the exact UBE rewards as proposed in the paper. We compute the POMBU rewards and
        then the gap term.
        """

        # POMBU rewards
        var_reward, next_state = self.ensemble_prediction(state, act)
        pombu_u = self.var_mean_value(next_state)

        # ===== Gap term =======
        next_state = next_state.repeat(1, self.act_n_samples, 1)
        next_action, _, _ = self.actor.sample(next_state)

        # We reshape the next_states to compute the mean value, i.e., have a single batch dimension
        # equal to num_members * state_batch
        state_batch_size = next_state.size(dim=1)
        next_state_mean_value = torch.reshape(
            next_state, (self.ensemble_size * state_batch_size, self.obs_dim)
        )
        next_action_mean_value = torch.reshape(
            next_action, (self.ensemble_size * state_batch_size, self.act_dim)
        )
        mean_value = self.get_mean_value(next_state_mean_value, next_action_mean_value)
        # Reshape back to 3D tensor
        mean_value = torch.reshape(
            mean_value, (self.ensemble_size, self.act_n_samples, self.batch_size)
        )

        # For Q^{p}(s', a') the input is directly `next_state, next_action`
        value = self.get_min_q(next_state, next_action, self.critic)
        # Reshape back to 3D tensor
        value = torch.reshape(value, (self.ensemble_size, self.act_n_samples, self.batch_size))

        # First we want to compute the variance over the stacked action dimensions
        if self.uncertainty_type == "exact_ube_2":
            gap_term = torch.var(value, dim=1) - torch.var(mean_value, dim=1)
        elif self.uncertainty_type == "exact_ube_3":
            gap_term = torch.var(value - mean_value, dim=1)

        # Then we average over the ensemble dimension
        gap_term = torch.mean(gap_term, dim=0)
        exact_ube_u = pombu_u - gap_term.unsqueeze(dim=-1)
        return var_reward, torch.clamp(exact_ube_u, min=0.0)

    def ensemble_prediction(self, state: torch.Tensor, act: torch.Tensor):
        """
        Compute next state and rewards from dynamics ensemble
        """
        dynamics_input = self.dynamics_model._get_model_input(state, act)

        # Get each model's predictions. We use the mean of the corresponding Gaussian (in principle
        # you could also sample from the Gaussian instead)
        pred_means, _ = self.dynamics_model.forward(dynamics_input)

        # Handle the case of a known reward function
        if self.reward_fn is None:
            next_state = pred_means[:, :, :-1]
            reward = torch.unsqueeze(pred_means[:, :, -1], dim=-1)
            var_reward = torch.var(reward, dim=0)
        else:
            next_state = pred_means
            var_reward = torch.zeros((self.batch_size, 1)).to(self.device)

        # If we are predicting deltas, then we need to add it to the pred samples
        if self.dynamics_model.target_is_delta:
            next_state += state.unsqueeze(dim=0)

        return var_reward, next_state

    def get_mean_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Pass the state-action batch through the ensemble of critics and aggregate the predictions
        using the mean.
        """
        return self.critic.mean_forward(self.get_critic_input(state, action))
