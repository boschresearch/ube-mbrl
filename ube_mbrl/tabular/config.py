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


default_parameters = dict(
    seed=0,
    deep_sea_size=5,
    num_episodes=10,
    transition_probability=0.95,
    opt_value=19.0,
    agent=dict(
        ensemble_size=5,
        gamma=0.99,
        agent_type="ofu",
        policy_type="greedy",
        boltzmann_temp=0.05,
        exploration_gain=1.0,
        uncertainty_type="exact_ube_3",
        ureward_min=0.0,
        model_type="normal",
        max_pi_steps=40,
        transition_repeat=1,
        model_prior=dict(mu=0.0, kappa=1.0, alpha=4.0, beta=4.0),
    ),
)
