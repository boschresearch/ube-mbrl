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

import numpy as np
from ube_mbrl.tabular.envs.tabular import TabularEnv


class DeepSea(TabularEnv):
    """
    Reference implementations:
    - Bsuite https://github.com/deepmind/bsuite/blob/main/bsuite/environments/deep_sea.py
    - https://github.com/stratisMarkou/sample-efficient-bayesian-rl/blob/master/code/Environments.py
    """

    def __init__(self, length: int):
        self.length = length
        self.row = 0
        self.col = 0
        self.state = 0
        self.move_cost = 0.01
        super().__init__(num_states=length * length, num_actions=2)
        self.p, self.r = self.get_mdp_representation()

    def step(self, action):
        reward = 0
        go_right = action == 1

        # Reward at the end of chain
        if self.col == self.length - 1 and go_right:
            reward += 1

        # Transition dynamics
        if go_right:
            self.col = np.clip(self.col + 1, 0, self.length - 1)
            reward -= self.move_cost / self.length
        else:
            self.col = np.clip(self.col - 1, 0, self.length - 1)

        # Always go down, no matter the action
        self.row += 1

        # termination
        if done := self.row == self.length:
            self.row = self.length - 1

        self.update_state()
        return self.state, reward, done, {}

    def update_state(self):
        self.state = self.row * self.length + self.col

    def reset(self):
        self.row = 0
        self.col = 0
        self.state = 0
        return self.state

    def render(self):
        pass

    def get_mdp_representation(self):
        """
        Returns the transition matrix P and the reward vector r. To be used as a debug tool in
        tabular RL.
        """
        # we consider an augmented state space with a terminal state.
        num_states = self.num_states + 1
        p = np.zeros((num_states, self.num_actions, num_states))
        r = np.zeros((num_states, self.num_actions))
        terminal = num_states - 1

        for s in range(num_states):
            row, col = self.get_rowcol(s)
            p[s, 0] = np.zeros(num_states)
            p[s, 1] = np.zeros(num_states)

            # Terminal state transition onto itself
            if s == terminal:
                p[s, 0, terminal] = 1
                p[s, 1, terminal] = 1

            # We don't case about upper-triangular states
            if col > row:
                p[s, 0, terminal] = 1
                p[s, 1, terminal] = 1
                continue

            # Rewarding state at bottom right corner
            if col == self.length - 1:
                r[s, 1] += 1

            if row < self.length - 1:
                new_row = row + 1
                new_col_left = np.clip(col - 1, 0, self.length - 1)
                new_col_right = np.clip(col + 1, 0, self.length - 1)
                new_s_left = self.get_coord(new_row, new_col_left)
                new_s_right = self.get_coord(new_row, new_col_right)
                p[s, 0, new_s_left] = 1
                p[s, 1, new_s_right] = 1
                r[s, 1] -= self.move_cost / self.length
            elif row == self.length - 1:
                # Go to terminal state
                p[s, 0, terminal] = 1
                p[s, 1, terminal] = 1

        return p, r

    def get_rowcol(self, s):
        row = s // self.length
        col = s % self.length
        return row, col

    def get_coord(self, row, col):
        return row * self.length + col


if __name__ == "__main__":
    length = 3
    env = DeepSea(length=length)
    env.get_mdp_representation()
    state = env.reset()
    done = False
    while not done:
        action = 1
        next_state, reward, done, _ = env.step(action)
        print(
            f"curr_state = {state}, action = {action}, next_state = {next_state}, reward = {reward}, done = {done}"
        )
        state = next_state
