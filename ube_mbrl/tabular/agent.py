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

from typing import Tuple

import numpy as np
from envs.tabular import TabularEnv
from ube_mbrl.mbrl.util import ReplayBuffer
from posterior import DirichletPosterior, NormalGammaPosterior, NormalPosterior


class TabularAgent:
    def __init__(self, env: TabularEnv, params: dict = None):
        self.env = env
        self.gamma = params["gamma"]
        self.boltzmann_temp = params["boltzmann_temp"]
        self.explore_gain = params["exploration_gain"]
        self.agent_type = params["agent_type"]
        self.uncertainty_type = params["uncertainty_type"]
        self.ensemble_size = params["ensemble_size"]
        self.policy_type = params["policy_type"]
        self.model_type = params["model_type"]
        self.ureward_min = params["ureward_min"]
        self.max_pi_steps = params["max_pi_steps"]

        # Adaptive transition repeat for DeepSea environment
        if hasattr(env, "length"):
            self.transition_repeat = self.env.length
        else:
            self.transition_repeat = params["transition_repeat"]

        # Add terminal state abstraction to the agent, not the environment
        self.num_states = env.observation_space.n + 1
        self.terminal = self.num_states - 1
        self.num_actions = env.action_space.n
        self.action_space = env.action_space

        self.counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        dirichlet_prior = env.observation_space.n ** (-0.5) * np.ones(
            (self.num_states, self.num_actions, self.num_states)
        )

        # Force terminal state transition onto itself
        epsilon = 1e-8
        dirichlet_prior[self.terminal, :, :] = epsilon * np.ones(
            (self.num_actions, self.num_states)
        )
        dirichlet_prior[self.terminal, :, self.terminal] = (1000) * np.ones((self.num_actions))
        self.p_posterior = DirichletPosterior(dirichlet_prior)

        if self.model_type == "normal_gamma":
            # Params for the normal-gamma model of the reward function r(s,a)
            model_prior = params["model_prior"]
            mu = model_prior["mu"]
            kappa = model_prior["kappa"]
            alpha = model_prior["alpha"]
            beta = model_prior["beta"]
            normalgamma_prior = np.array(
                [[[mu, kappa, alpha, beta]] * self.num_actions] * self.num_states
            )
            # Set specific params for terminal state, which should always be close to zero when sampling
            normalgamma_prior[self.terminal, :] = [0.0, 1e9, 1e12, 1e9]
            self.r_posterior = NormalGammaPosterior(normalgamma_prior)
        elif self.model_type == "normal":
            mu = 0
            tau = 1
            normal_prior = np.array([[[mu, tau]] * self.num_actions] * self.num_states)
            # Force terminal state to be close-to-deterministic with zero reward
            normal_prior[self.terminal, :] = [0.0, 1e9]
            self.r_posterior = NormalPosterior(normal_prior)

        # Policy is represented as a matrix (table) of size SxA. We initialize it with some
        # arbitrary but deterministic action selection
        random_actions = np.random.choice(self.num_actions, self.num_states)
        self.pi = self.actions_to_policy_matrix(random_actions)

    def update(self, ep_buffer: ReplayBuffer, t: int):
        """
        Update the dynamics and value functions / policy based on the data in the buffer
        """

        self.update_posterior_mdp(ep_buffer)

        if self.agent_type == "psrl":
            # Sample from posterior
            p, r = self.sample_mdp_from_posterior()

            # Solve MDP to update policy
            self.pi = self.solve_mdp(p, r)

        elif self.agent_type == "ofu":
            # Get the mean MDP from the updated posterior
            p_bar, r_bar = self.get_mean_mdp()

            # Solve MDP with uncertainty quantification
            self.pi = self.solve_mdp(p_bar, r_bar)

    def solve_mdp(self, p, r) -> np.ndarray:
        """
        Solves MDP specified by transition function P and reward function r by policy iteration.

        Returns the optimal policy found by PI
        """
        # Init policy
        random_actions = np.random.choice(self.num_actions, self.num_states)
        pi = self.actions_to_policy_matrix(random_actions)

        if self.agent_type == "ofu":
            # Sample models from the posterior to compose an ensemble of MDPs
            ensemble = self.sample_ensemble_from_posterior(num_samples=self.ensemble_size)

        for _ in range(self.max_pi_steps):
            # Policy evaluation
            v, q = self.solve_bellman_eq(p, r, pi, discount=self.gamma)

            if self.agent_type == "ofu":
                self.mean_qf = q
                self.pi = pi
                u_q = self.compute_q_uncertainty(p, ensemble)
                q += self.explore_gain * np.sqrt(u_q.clip(min=0.0))

            # Policy improvement
            new_pi = self.policy_improvement(q)

            # If policy doesn't change, algorithm has converged
            if np.prod(new_pi == pi) == 1:
                break
            else:
                pi = new_pi

        return pi

    def compute_q_uncertainty(self, p_bar: np.ndarray, ensemble: tuple) -> np.ndarray:
        if self.uncertainty_type == "none":
            u_qf = np.array([0.0])
        elif self.uncertainty_type == "ensemble":
            self.u_vf, u_qf = self.compute_ensemble_variance(ensemble)
        else:
            uncertainty_rewards = self.get_uncertainty_rewards(p_bar, ensemble).clip(
                min=self.ureward_min
            )
            self.u_vf, u_qf = self.solve_bellman_eq(
                p_bar, uncertainty_rewards, self.pi, discount=self.gamma**2
            )
        return u_qf

    def update_posterior_mdp(self, ep_buffer: ReplayBuffer):
        """
        Updates the posterior distribution over MDPs:
        1) Update the Dirichlet distribution for the transition probabilities p(s'|s,a)
        2) Update the Normal-Gamma (or Normal) model of the mean reward function r(s,a)
        """
        counts = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_sums = np.zeros((self.num_states, self.num_actions))
        rsquared_sums = np.zeros((self.num_states, self.num_actions))
        data = ep_buffer.get_all()

        for transition in data:
            s, a, s_prime, r, done = transition.astuple()
            if done:
                # artificially transition from s_prime to terminal
                counts[s, a, self.terminal] += self.transition_repeat
            else:
                counts[s, a, s_prime] += self.transition_repeat

            r_sums[s, a] += r * self.transition_repeat
            rsquared_sums[s, a] += (r**2) * self.transition_repeat

        # Update MDP posterior
        self.p_posterior.update_params(counts)
        self.r_posterior.update_params(counts, r_sums, rsquared_sums)

    def policy_improvement(self, qf: np.ndarray) -> np.ndarray:
        """
        Compute the greedy or boltzmann policy according to some utility function qf
        """
        if self.policy_type == "greedy":
            greedy_actions = np.array(
                [np.random.choice(np.argwhere(q == np.amax(q))[0]) for q in qf]
            )
            pi = self.actions_to_policy_matrix(greedy_actions)
        elif self.policy_type == "boltzmann":
            pi = np.exp(qf / self.boltzmann_temp)
            pi = pi / pi.sum(axis=1)[:, None]

        return pi

    def get_uncertainty_rewards(self, p_bar: np.ndarray, ensemble: tuple) -> np.ndarray:
        if self.uncertainty_type == "pombu":
            return self.compute_pombu_rewards(ensemble)
        if self.uncertainty_type in ("exact_ube_1", "exact_ube_2", "exact_ube_3"):
            return self.compute_exact_ube_rewards(p_bar, ensemble)
        else:
            NotImplementedError(f"Uncertainty type {self.uncertainty_type} is not implemented")

    def solve_bellman_eq(
        self, dynamics: np.ndarray, rewards: np.ndarray, policy: np.ndarray, discount: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves a Bellman equation in closed form for tabular RL problems. Allows to solve both the
        classical Bellman equation for value functions, as well as UBEs for the uncertainty of
        values
        """
        r_pi = np.einsum("ij, ij -> i", rewards, policy)
        p_pi = np.einsum("ijk, ij -> ik", dynamics, policy)
        vf = np.linalg.solve(np.eye(self.num_states) - discount * p_pi, r_pi)
        qf = rewards + discount * np.einsum("ijk, k -> ij", dynamics, vf)
        return vf, qf

    def get_mean_mdp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the mean MDP from the transition and reward distributions.
        1) For a Dirichlet distribution, the mean is the alpha counts divided by their sum
        2) For the Normal-Gamma, we want the mean, which is the "mu" parameter.
        """
        P_bar = self.p_posterior.get_mean()
        r_bar = self.r_posterior.get_mean()
        return P_bar, r_bar

    def sample_ensemble_from_posterior(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Composes an ensemble of MDPs of size num_samples.
        The resulting model has dimensions: (ensemble x S x A x S)
        """
        p_ensemble = []
        r_ensemble = []
        for _ in range(num_samples):
            p, r = self.sample_mdp_from_posterior()
            p_ensemble.append(p)
            r_ensemble.append(r)
        return np.stack(p_ensemble, axis=0), np.stack(r_ensemble, axis=0)

    def sample_mdp_from_posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples a dynamics model p(s' | s,a) and reward function r(s,a) from their respective
        distributions
        """
        p = np.zeros((self.num_states, self.num_actions, self.num_states))
        r = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                p[s, a, :] = self.p_posterior.sample(s, a)
                r[s, a] = self.r_posterior.sample(s, a)

        return p, r

    def compute_ensemble_variance(self, ensemble: tuple) -> np.ndarray:
        """
        From an ensemble of transition models, compute the corresponding Q-functions for each and
        return the empirical variance of the values of the ensemble.
        """
        v_ensemble, q_ensemble = self.compute_value_ensemble(ensemble)
        return np.var(v_ensemble, axis=0), np.var(q_ensemble, axis=0)

    def compute_pombu_rewards(self, ensemble: tuple) -> np.ndarray:
        """
        Given samples from the posterior over MDPs, computes an estimate of u_t(s,a), the notion
        of local uncertainties from the POMBU paper.

        In the einsum:
            - 'e' refers to the ensemble dimension
            - 'ij' refers to (s,a)
            - 'kl' to (s', a')
        """
        p, r = ensemble
        reward_var = np.var(r, axis=0)
        expected_next_q = np.einsum("eijk, kl, kl -> eij", p, self.pi, self.mean_qf, optimize=True)
        return reward_var + np.var(expected_next_q, axis=0) * self.gamma**2

    def compute_exact_ube_rewards(self, p_bar: np.ndarray, ensemble: np.ndarray) -> np.ndarray:
        """
        Compute the uncertainty rewards leading to an UBE whose solution represent the uncertainty
        about the Q-function, wrt the current posterior distribution over models.
        """
        p_ensemble, r_ensemble = ensemble
        # First, compute the q-values for each member of the ensemble
        _, q_ensemble = self.compute_value_ensemble(ensemble)

        # First Option
        if self.uncertainty_type == "exact_ube_1":
            reward_var = np.var(r_ensemble, axis=0)
            var_mean_value = self.compute_variance_next_qvalue(p_bar, self.mean_qf)
            var_ensemble_value = self.compute_variance_next_qvalue(p_ensemble, q_ensemble)
            gap = np.mean(var_ensemble_value, axis=0)
            return reward_var + (var_mean_value - gap) * self.gamma**2
        else:
            pombu_rewards = self.compute_pombu_rewards(ensemble)
            # Second option:
            if self.uncertainty_type == "exact_ube_2":
                var_ensemble_value = self.compute_variance_next_qvalue(p_ensemble, q_ensemble)
                var_mean_value = self.compute_variance_next_qvalue(p_ensemble, self.mean_qf)
                pombu_gap = np.mean(var_ensemble_value - var_mean_value, axis=0)
            # Third option:
            elif self.uncertainty_type == "exact_ube_3":
                var_diff_value = self.compute_variance_next_qvalue(
                    p_ensemble, q_ensemble - self.mean_qf
                )
                pombu_gap = np.mean(var_diff_value, axis=0)

            return pombu_rewards - pombu_gap * self.gamma**2

    def compute_variance_next_qvalue(self, p: np.ndarray, qvalue: np.ndarray) -> np.ndarray:
        """
        Computes the variance of the next Q-values, Q(s', a'), where s' ~ p(. |s,a) and
        a' ~ pi(.|s'). Note that p is actually an ensemble of models, so we compute the variance for
        each model individually.
        """
        # First case is where the p model is not an ensemble
        if p.ndim == 3:
            second_moment = np.einsum("ijk, kl, kl -> ij", p, self.pi, qvalue**2, optimize=True)
            sqr_first_moment = (
                np.einsum("ijk, kl, kl -> ij", p, self.pi, qvalue, optimize=True) ** 2
            )
        # Compute variance using expectation definition
        elif qvalue.ndim == 3:
            second_moment = np.einsum(
                "eijk, kl, ekl -> eij", p, self.pi, qvalue**2, optimize=True
            )
            sqr_first_moment = (
                np.einsum("eijk, kl, ekl -> eij", p, self.pi, qvalue, optimize=True) ** 2
            )
        else:
            second_moment = np.einsum("eijk, kl, kl -> eij", p, self.pi, qvalue**2, optimize=True)
            sqr_first_moment = (
                np.einsum("eijk, kl, kl -> eij", p, self.pi, qvalue, optimize=True) ** 2
            )
        return second_moment - sqr_first_moment

    def compute_value_ensemble(self, ensemble: tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns an ensemble of V-functions and Q-functions of the same size as the ensemble of
        transition models passed as argument to the function.
        """
        p_ensemble, r_ensemble = ensemble
        num_models = p_ensemble.shape[0]
        vfs, qfs = [], []
        for i in range(num_models):
            p = p_ensemble[i]
            r = r_ensemble[i]
            vf, qf = self.solve_bellman_eq(p, r, self.pi, self.gamma)
            vfs.append(vf)
            qfs.append(qf)
        v_ensemble = np.stack([value for value in vfs], axis=0)
        q_ensemble = np.stack([q_value for q_value in qfs], axis=0)
        return v_ensemble, q_ensemble

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Given an observation, returns an action sampled from the policy. In general, we assume the
        policy to be stochastic
        """
        return np.random.choice(self.num_actions, p=self.pi[obs])

    def actions_to_policy_matrix(self, actions: np.ndarray) -> np.ndarray:
        """
        Transforms deterministic action selection into the proper (probabilistic) policy matrix form
        """
        pi = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            pi[s, actions[s]] = 1
        return pi
