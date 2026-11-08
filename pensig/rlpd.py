"""SAC with demonstrations in the replay buffer, as in RLPD (Ball et al. 2023): half of every batch comes from
the scripted stack's episodes, half from the policy's own, and the critic has LayerNorm so its values stay
bounded on actions the data does not cover.

Why not PPO: on the tower curriculum PPO learned the grasp phases to ~50 %, then lost them while the later
phases stayed above 80 %, and its action std fell to 0.07 (tower_cur_41021443). SAC keeps its entropy at a
target, and the replay buffer keeps the grasp episodes it would otherwise forget.
"""

import torch
from torch import nn
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.sac.policies import SACPolicy


def _mlp_ln(n_in: int, arch: list[int]) -> nn.Sequential:
    layers, d = [], n_in
    for h in arch:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 1))


class LNSACPolicy(SACPolicy):
    """SACPolicy whose Q networks have LayerNorm after every hidden layer."""

    def make_critic(self, features_extractor=None):
        critic = super().make_critic(features_extractor)
        n_in = critic.q_networks[0][0].in_features  # features + action
        arch = self.critic_kwargs["net_arch"]
        for i in range(critic.n_critics):
            q = _mlp_ln(n_in, arch)
            critic.q_networks[i] = q
            critic.add_module(f"qf{i}", q)
        return critic.to(self.device)


class DemoMixBuffer(ReplayBuffer):
    """The online replay buffer, plus a fixed demonstration buffer that supplies `demo_ratio` of every batch."""

    def __init__(self, *args, demo_ratio: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_ratio = demo_ratio
        self.demo = None

    def set_demos(self, obs, actions, rewards, next_obs, dones):
        n = len(obs)
        self.demo = ReplayBuffer(n, self.observation_space, self.action_space, device=self.device, n_envs=1,
                                 handle_timeout_termination=False)
        self.demo.observations[:, 0] = obs
        self.demo.next_observations[:, 0] = next_obs
        self.demo.actions[:, 0] = actions
        self.demo.rewards[:, 0] = rewards
        self.demo.dones[:, 0] = dones
        self.demo.full, self.demo.pos = True, 0

    def sample(self, batch_size: int, env=None):
        n_demo = int(batch_size * self.demo_ratio) if self.demo is not None else 0
        own = super().sample(batch_size - n_demo, env)
        if n_demo == 0:
            return own
        demo = self.demo.sample(n_demo, env)
        return type(own)(*[None if a is None else torch.cat([a, b]) for a, b in zip(own, demo)])
