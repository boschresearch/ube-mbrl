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
    env_name="SparsePendulum-v0",
    action_cost=0.2,
    noise_std=0.0,
    num_steps=30e3,
    num_model_rollouts_per_step=400,
    model_rollout_length=10,
    freq_model_retrain=400,
    num_updates_to_retain_buffer=1,
    agent_updates_per_step=20,
    buffer_init_steps=1000,
    freq_agent_eval=400,
    steps_per_epoch=400,
    eval_episodes=1,
    use_mean_model_buffer=False,
    learned_rewards=True,
    env_buffer_capacity=30e3,
    model_train=dict(
        learning_rate=3e-4,
        weight_decay=5e-5,
        batch_size=256,
        validation_ratio=0.05,
        patience=1,
        num_epochs=10,
    ),
    agent=dict(
        ensemble_size=5,
        critics_per_model=1,
        gamma=0.99,
        alpha_temp=0.2,
        smoothness_coef=0.005,
        auto_entropy_tuning=True,
        clip_grad_norm=False,
        batch_size=256,
        target_update_freq=1,
        actor_update_freq=1,
        use_ube_target=True,
        uncertainty_penalty=1,
        uncertainty_type="exact_ube_3",
        rollout_mode="random_model",
        actor=dict(
            num_layers=2,
            hid_size=64,
            activation_fn=dict(_target_="torch.nn.Tanh"),
            learning_rate=3e-4,
        ),
        critic=dict(
            num_layers=2,
            hid_size=256,
            activation_fn=dict(_target_="torch.nn.Tanh"),
            learning_rate=3e-4,
        ),
        ube=dict(
            act_n_samples=10,
            num_layers=2,
            hid_size=256,
            output_fn="softplus",
            activation_fn=dict(_target_="torch.nn.Tanh"),
            learning_rate=3e-4,
            regularization_penalty=5.0,
        ),
    ),
)
