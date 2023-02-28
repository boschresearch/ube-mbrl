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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.pyplot import cm
from ube_mbrl.mbrl.util.replay_buffer import ReplayBuffer
from torch.functional import cartesian_prod
from ube_mbrl.agent.qusac import QUSAC


class PendulumPlotter:
    def __init__(self, device, num_angles=30, num_vels=30):
        # state-space limits for plotting
        self.max_angle = np.pi
        self.max_vel = 10
        self.num_angles = num_angles
        self.num_vels = num_vels

        # Discretized velocities and angles
        self.disc_angles = np.linspace(-self.max_angle, self.max_angle, num_angles)
        self.disc_vels = np.linspace(-self.max_vel, self.max_vel, self.num_vels)

        # Inputs to be fed into NN -- either value function or UBENet
        self.inputs = self.build_function_inputs(self.disc_angles, self.disc_vels).to(device)
        self.xaxis, self.yaxis = np.meshgrid(self.disc_angles, self.disc_vels)

    def plot_values(self, dir: str, agent: QUSAC, ep: int):
        # Get the mean actions for all inputs
        mean_actions, log_std = agent.actor.forward(self.inputs)
        std_actions = log_std.exp()
        values = agent.get_min_q(self.inputs, mean_actions, agent.critic)
        mean_values = torch.mean(values, dim=0)
        ensemble_std_values = torch.std(values, dim=0)
        if hasattr(agent, "uncertainty"):
            ube_std_values = torch.sqrt(
                agent.uncertainty(agent.get_critic_input(self.inputs, mean_actions))
            )
            self.plot_function_contour(
                ube_std_values,
                f"UBE std Values - Episode {ep}",
                dir,
                f"ube_values_ep{ep}",
                colormap=cm.viridis,
            )
        self.plot_function_contour(
            mean_values, f"Mean Values - Episode {ep}", dir, f"mean_values_ep{ep}"
        )
        self.plot_function_contour(
            ensemble_std_values,
            f"Ensemble std Values - Episode {ep}",
            dir,
            f"ensemble_std_ep{ep}",
            colormap=cm.viridis,
        )
        self.plot_function_contour(
            mean_actions, f"Mean Actions - Episode {ep}", dir, f"mean_actions_ep{ep}"
        )
        self.plot_function_contour(
            std_actions,
            f"Std Actions - Episode {ep}",
            dir,
            f"std_actions_ep{ep}",
            colormap=cm.viridis,
        )

    def plot_buffer(self, dir: str, buffer: ReplayBuffer, ep: int, fname: str):
        num_stored = buffer.num_stored

        # Extract all the states from replay buffer
        cos_theta = buffer.obs[: num_stored - 1, 0]
        sin_theta = buffer.obs[: num_stored - 1, 1]
        theta_dot = buffer.obs[: num_stored - 1, 2]
        theta = np.arctan2(sin_theta, cos_theta)

        # Second plot
        counts, yedges, xedges = np.histogram2d(theta_dot, theta, bins=100)
        xcenter = np.convolve(xedges, np.ones(2), "valid") / 2
        ycenter = np.convolve(yedges, np.ones(2), "valid") / 2

        X, Y = np.meshgrid(xcenter, ycenter)
        fig, ax = plt.subplots()
        num_levels = 100
        levels = np.linspace(0.5, counts.max(), num_levels)
        data = self.fix_white_lines(ax.contourf(X, Y, counts, cmap=cm.coolwarm, levels=levels))
        plt.colorbar(data, ax=ax)
        ax.set_ylim(bottom=-self.max_vel, top=self.max_vel)
        fig.suptitle(f"{fname} Visitation Count - Episode {ep}", fontsize=20)
        self.add_xy_labels(ax)
        self.save_fig(dir, f"{fname}_visit_counts_ep{ep}")
        plt.close(fig)

    def plot_function_contour(
        self,
        values: torch.Tensor,
        plt_title: str,
        save_dir: str,
        fig_fname: Optional[str] = None,
        colormap: Optional[Union[str, Colormap]] = cm.coolwarm,
    ):
        fig, ax = plt.subplots()
        num_levels = 100
        values = torch.reshape(values, (self.num_angles, self.num_vels)).detach().cpu().numpy()
        data = self.fix_white_lines(
            ax.contourf(self.xaxis, self.yaxis, values, num_levels, cmap=colormap)
        )
        plt.colorbar(data, ax=ax)
        fig.suptitle(plt_title, fontsize=20)
        self.add_xy_labels(ax)
        self.save_fig(save_dir, fig_fname)
        plt.close(fig)

    def build_function_inputs(self, disc_angles, disc_vels, ensemble_size=1):
        angles = torch.Tensor(disc_angles)
        vels = torch.Tensor(disc_vels)
        inputs = cartesian_prod(angles, vels)
        theta, theta_dot = torch.tensor_split(inputs, 2, dim=-1)
        inputs = torch.cat([torch.cos(theta), torch.sin(theta), theta_dot], dim=-1)
        # return inputs.repeat(ensemble_size, 1, 1)
        return inputs

    def add_xy_labels(self, ax: Axes):
        ax.set_xlabel("Angle [rad]")
        ax.set_ylabel("Angular Velocity [rad/s]")

    def save_fig(self, dir: str, save_fname: str):
        fig_dir = dir + f"/{save_fname}.pdf"
        plt.savefig(fig_dir, bbox_inches="tight", transparent=False)

    def fix_white_lines(self, cnt):
        for c in cnt.collections:
            c.set_edgecolor("face")
        return cnt
