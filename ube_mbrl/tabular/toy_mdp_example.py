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

from itertools import product
from typing import Callable

import numpy as np

from envs.toy_mdp import ToyMDP


def run():
    """
    Variance estimation in the toy MRP example from the paper (Fig. 1)

    We compare the ground-truth variance estimated directly from the (finite) samples of the
    posterior and the estimates coming from POMBU and exact UBE.

    This serves as a sanity-check that indeed UBE obtains the ground-truth variance and POMBU an
    upper-bound.

    NOTE: both options for UBE result in the same variance estimate.
    """
    alphas, betas = get_fixed_size_posterior()
    p_post, r_post = build_mdp_posterior(alphas, betas, is_correlated=False, mix_fn=product)
    v_post = build_value_posterior(p_post, r_post)

    ensemble_var = np.var(v_post, axis=0)
    pombu_var = compute_ube_var(p_post, r_post, v_post, type="pombu")
    exact_pombu_var = compute_ube_var(p_post, r_post, v_post, type="exact_pombu")

    print(f"Mean values = {np.mean(v_post, axis=0)}")
    print(f"Ground-truth variance = {ensemble_var}")
    print(f"POMBU variance = {pombu_var}")
    print(f"Exact UBE variance = {exact_pombu_var}")


def get_fixed_size_posterior():
    alphas = [0.7, 0.6]
    betas = [0.5, 0.4]
    return alphas, betas


def solve_bellman_equation(p: np.ndarray, r: np.ndarray, discount: float = 0.99):
    return np.linalg.inv(np.eye(p.shape[0]) - discount * p).dot(r)


def compute_ube_var(
    p_post: np.ndarray,
    r_post: np.ndarray,
    v_post: np.ndarray,
    type: str,
    discount: float = 0.99,
):
    if type == "pombu":
        u = compute_pombu_rewards(p_post, r_post, v_post, discount)
    elif type == "exact_pombu":
        u = compute_exact_ube_rewards(p_post, r_post, v_post, discount)
    else:
        NotImplementedError()

    return solve_bellman_equation(p=np.mean(p_post, axis=0), r=u, discount=discount**2)


def compute_pombu_rewards(
    p_post: np.ndarray, r_post: np.ndarray, v_post: np.ndarray, discount: float
) -> np.ndarray:
    reward_var = np.var(r_post, axis=0)
    mean_v = np.mean(v_post, axis=0)
    expected_next_v = np.einsum("eij, j -> ei", p_post, mean_v)
    return reward_var + np.var(expected_next_v, axis=0) * discount**2


def compute_exact_ube_rewards(
    p_post: np.ndarray, r_post: np.ndarray, v_post: np.ndarray, discount: float
) -> np.ndarray:
    pombu_rewards = compute_pombu_rewards(p_post, r_post, v_post, discount)
    mean_v = np.mean(v_post, axis=0)

    # Option 1:
    diff_value = v_post - mean_v
    var_diff_value = compute_variance_next_value(p_post, diff_value)
    pombu_gap = np.mean(var_diff_value, axis=0)

    # Option 2:
    # var_ensemble_value = compute_variance_next_value(p_post, v_post)
    # var_mean_value = compute_variance_next_value(p_post, mean_v)
    # pombu_gap = np.mean(var_ensemble_value - var_mean_value, axis=0)

    return pombu_rewards - pombu_gap * discount**2


def compute_variance_next_value(p: np.ndarray, vf: np.ndarray) -> np.ndarray:
    if vf.ndim == 2:
        second_moment = np.einsum("eij, ej -> ei", p, vf**2)
        sqr_first_moment = np.einsum("eij, ej -> ei", p, vf) ** 2
    else:
        second_moment = np.einsum("eij, j -> ei", p, vf**2)
        sqr_first_moment = np.einsum("eij, j -> ei", p, vf) ** 2
    return second_moment - sqr_first_moment


def build_mdp_posterior(alphas, betas, is_correlated, mix_fn: Callable):
    p = []
    r = []
    for alpha, beta in mix_fn(alphas, betas):
        mdp = ToyMDP(alpha, beta, is_correlated)
        p.append(mdp.p)
        r.append(mdp.r)
    return np.array(p), np.array(r)


def build_value_posterior(p_post: np.ndarray, r_post: np.ndarray) -> np.ndarray:
    return np.array([solve_bellman_equation(p, r) for p, r in zip(p_post, r_post)])


if __name__ == "__main__":
    run()
