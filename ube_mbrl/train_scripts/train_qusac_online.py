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

import os

import pybullet_envs
import torch
import ube_mbrl.envs
import ube_mbrl.mbrl.models as models
import ube_mbrl.mbrl.util.common as mbrl_utils
import ube_mbrl.utils.common as utils_common
import ube_mbrl.utils.data_collection as utils_data
from hydra.experimental import compose, initialize
from ube_mbrl.agent import QUSAC
from ube_mbrl.conf import qusac_online_default_params
from ube_mbrl.envs.plot import PendulumPlotter
from ube_mbrl.mbrl.util import ReplayBuffer
from ube_mbrl.utils.video import close_virtual_display


def train(parameters):
    env_name, env, eval_env = utils_common.get_envs_from_name(parameters["env_name"], parameters)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # Fix RNG
    seed = parameters["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng, generator = utils_common.fix_rng(env, eval_env=eval_env, seed=seed, device=device)

    # Dynamics model
    initialize(config_path="../conf")
    cfg = compose(config_name="mbrl_lib_config.yaml")
    cfg["algorithm"]["learned_rewards"] = parameters["learned_rewards"]
    cfg["dynamics_model"]["ensemble_size"] = parameters["agent"]["ensemble_size"]
    utils_common.set_device_in_hydra_cfg(device, cfg)
    dynamics_model = mbrl_utils.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    env_buffer_capacity = int(parameters["env_buffer_capacity"])
    env_replay_buffer = ReplayBuffer(env_buffer_capacity, obs_shape, act_shape, rng=rng)
    model_lr = parameters["model_train"]["learning_rate"]
    model_wd = parameters["model_train"]["weight_decay"]
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=model_lr, weight_decay=model_wd)

    # Create gym-like wrapper to rollout the model
    term_fn = utils_common.get_term_fn(env_name)
    reward_fn = None if parameters["learned_rewards"] else utils_common.get_reward_fn(env_name)
    ensemble_envs = utils_common.ensemble_to_envs(
        dynamics_model,
        env,
        generator,
        reward_fn=reward_fn,
        termination_fn=term_fn,
        add_mean_model=True,
    )
    num_model_rollouts_per_step = parameters["num_model_rollouts_per_step"]
    model_rollout_length = parameters["model_rollout_length"]

    # Define buffer(s) to store model-based rollouts
    freq_model_retrain = parameters["freq_model_retrain"]
    rollout_batch_size = num_model_rollouts_per_step * freq_model_retrain
    num_updates_to_retain_buffer = parameters["num_updates_to_retain_buffer"]
    model_buffers_capacity = (
        model_rollout_length * rollout_batch_size * num_updates_to_retain_buffer
    )
    model_buffers = [
        ReplayBuffer(model_buffers_capacity, obs_shape, act_shape, rng=rng)
        for _ in range(len(dynamics_model) + 1)
    ]

    # Instantiate the QUSAC agent that will be trained
    device = torch.device(device)
    agent = QUSAC(env, device, parameters["agent"], dynamics_model, reward_fn=reward_fn)

    # Pre-fill the env replay buffer with data from a random agent
    buffer_init_steps = parameters["buffer_init_steps"]
    mbrl_utils.rollout_agent_trajectories(
        env, buffer_init_steps, agent, {}, replay_buffer=env_replay_buffer
    )

    # Directory for creating agent checkpoints
    artifact_dir = os.getcwd()
    checkpoints_dir = os.path.join(artifact_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # For the pendulum we have more in-depth debugging
    if env_name == "SparsePendulum-v0":
        plotter = PendulumPlotter(device=device)

    # Agent training loop
    num_steps = int(parameters["num_steps"])
    steps_per_epoch = int(parameters["steps_per_epoch"])
    global_step = 0
    agent_updates = 1
    while global_step <= num_steps:
        obs, done = env.reset(), False
        ep_step = 0
        while not (done or ep_step >= steps_per_epoch):
            # Take an env step
            next_obs, _, done, _ = utils_common.step_env_and_add_to_buffer(
                env, obs, agent, {}, env_replay_buffer
            )

            # Retrain model and collect new data after desired number of steps
            if global_step % parameters["freq_model_retrain"] == 0:
                utils_common.train_model(
                    dynamics_model, model_trainer, env_replay_buffer, parameters["model_train"]
                )
                utils_data.collect_ensemble_model_transitions(
                    ensemble_envs,
                    agent,
                    model_buffers,
                    env_replay_buffer,
                    rollout_length=model_rollout_length,
                    batch_size=rollout_batch_size,
                    use_mean_model_buffer=parameters["use_mean_model_buffer"],
                    rollout_mode=parameters["agent"]["rollout_mode"],
                )

            for _ in range(parameters["agent_updates_per_step"]):
                agent.update_params(model_buffers, agent_updates)
                agent_updates += 1

            # Evaluate agent
            if (global_step + 1) % parameters["freq_agent_eval"] == 0:
                epoch_dir = os.path.join(checkpoints_dir, f"step_{global_step+1}")
                os.makedirs(epoch_dir, exist_ok=True)
                eval_dict = utils_common.evaluate(
                    agent,
                    eval_env,
                    epoch_dir,
                    num_episodes=parameters["eval_episodes"],
                    max_steps=steps_per_epoch,
                )
                ep = global_step // steps_per_epoch
                print(f"Episode {ep} return = {eval_dict['avg_return']}")
                if env_name == "SparsePendulum-v0":
                    plotter.plot_values(epoch_dir, agent, ep=ep)
                    plotter.plot_buffer(epoch_dir, env_replay_buffer, ep=ep, fname="Env")
                    plotter.plot_buffer(epoch_dir, model_buffers[-1], ep=ep, fname="MeanModel")
                    plotter.plot_buffer(epoch_dir, model_buffers[0], ep=ep, fname="Model0")
                agent.save(epoch_dir)
                dynamics_model.deep_save(epoch_dir)

            obs = next_obs
            ep_step += 1
            global_step += 1
            if global_step > num_steps:
                break

    # Close env and pyvirtual display before exiting
    env.close()
    eval_env.close()
    close_virtual_display()


if __name__ == "__main__":
    train(qusac_online_default_params)
