"""SAC with demonstrations in the replay buffer, as in RLPD (Ball et al. 2023): half of every batch comes from
the scripted stack's episodes, half from the policy's own, and the critic has LayerNorm so its values stay
bounded on actions the data does not cover.

Why not PPO: on the tower curriculum PPO learned the grasp phases to ~50 %, then lost them while the later
phases stayed above 80 %, and its action std fell to 0.07 (tower_cur_41021443). SAC keeps its entropy at a
target, and the replay buffer keeps the grasp episodes it would otherwise forget.

Demonstrations in the buffer were not enough (tower_sac_41035915): from 1M to 2M steps the deterministic policy
went from 27/40 to 1/40 started at the grasp, and near the block it moved away and waited for the time limit.
A protective stop costs 3 at once, the time limit 3 about 150 steps later, and the critic values the noisy
policy, which near the block stops often. The critic cannot fix that from the demonstrations alone: their next
actions come from the policy. DemoSAC adds a behaviour cloning term on the demonstration half of every batch
(DDPGfD, Vecerik et al. 2017; Nair et al. 2018; TD3+BC, Fujimoto and Gu 2021, for the Q scaling).
The Q filter of Nair et al. is an option, off by default: it keeps the term only where the critic prefers the
script's action, and here the critic is what prefers moving away.

The gripper (tower_sac_41085727, 1M steps): from the descent onto the tower, sampled actions placed 30/30 and the
mean action 2/30. The mean reached the place height and stayed there closed, though the critic valued opening
higher at every step (8.8 vs 8.2). Two things held it: cloning pulled it towards closed (at that height the
script still descends, closed, until it feels contact), and the gripper's std sat at its maximum (7.4 before tanh),
so the SAC loss, an average over samples that open about half the time, barely depends on the mean. So after the
pre-training the cloning term leaves the gripper out (`bc_grip`), and the gripper's std is capped at 1 before tanh
(GripActor): its sampled command then follows the mean, and the SAC loss moves the mean where the critic prefers.
Maximising Q at the mean action as well (`det_weight`, the DDPG/TD3 gradient) is an option, off by default: in a
60k-step fine-tune from that checkpoint it broke every phase (0/20 even from the plain reset).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, SACPolicy

GRIP_LOG_STD_MAX = 0.0  # std 1 before tanh for the gripper; SB3's cap (2, std 7.4) for the motion


def _mlp_ln(n_in: int, arch: list[int]) -> nn.Sequential:
    layers, d = [], n_in
    for h in arch:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 1))


class GripActor(Actor):
    """SB3's SAC actor, with the gripper's (last action) log std capped at GRIP_LOG_STD_MAX."""

    def get_action_dist_params(self, obs):
        mean, log_std, kw = super().get_action_dist_params(obs)
        cap = torch.full_like(log_std[:1], float("inf"))
        cap[:, -1] = GRIP_LOG_STD_MAX
        return mean, torch.minimum(log_std, cap), kw


class LNSACPolicy(SACPolicy):
    """SACPolicy whose Q networks have LayerNorm after every hidden layer, and whose actor is a GripActor."""

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return GripActor(**actor_kwargs).to(self.device)

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
        self.last_n_demo = 0  # demonstration rows at the end of the last batch

    def set_demos(self, obs, actions, rewards, next_obs, dones, clean=None):
        """`clean`: the actions to clone, by default the ones played."""
        n = len(obs)
        self.demo_clean = torch.as_tensor(actions if clean is None else clean, dtype=torch.float32)
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
        self.last_n_demo = n_demo
        if n_demo == 0:
            return own
        demo = self.sample_demos(n_demo, env)
        return type(own)(*[None if a is None else torch.cat([a, b]) for a, b in zip(own, demo)])

    def sample_demos(self, n: int, env=None):
        """n demonstration transitions; their actions to clone in self.last_clean."""
        idx = np.random.randint(0, self.demo.buffer_size, size=n)
        self.last_clean = self.demo_clean[idx].to(self.device)
        return self.demo._get_samples(idx, env)


class DemoSAC(SAC):
    """SAC whose actor loss also pulls its mean action towards the demonstrated one on the demonstration rows of
    each batch (DemoMixBuffer puts them last). The SAC part is divided by the batch's mean |Q|, as in TD3+BC, so
    that `bc_weight` means the same whatever the scale of the returns."""

    def __init__(self, *args, bc_weight: float = 1.0, q_filter: bool = False, bc_grip: float = 0.0,
                 det_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_weight = bc_weight
        self.q_filter = q_filter
        self.bc_grip = bc_grip  # weight of the gripper in the cloning term during training (pre-training: 1)
        self.det_weight = det_weight  # weight of -Q(s, mean action) in the actor loss
        self.bc_stats = {"bc": [], "kept": []}

    def bc_term(self, obs, actions, grip: float = 1.0):
        """Mean squared distance from the actor's mean action to the demonstrated one, and the share of rows
        kept by the Q filter. The gripper target is +-0.9: binary_grip treats anything beyond 0.5 the same, and
        +-1 would push tanh into saturation."""
        target = actions.clone()
        target[:, 4] = target[:, 4].clamp(-0.9, 0.9)
        mu = self.actor(obs, deterministic=True)
        w = torch.ones(mu.shape[1], device=mu.device)
        w[4] = grip
        err = ((mu - target) ** 2 * w).sum(dim=1, keepdim=True)
        if self.q_filter:
            with torch.no_grad():
                q_demo = torch.min(torch.cat(self.critic(obs, actions), dim=1), dim=1, keepdim=True)[0]
                q_mu = torch.min(torch.cat(self.critic(obs, mu.detach()), dim=1), dim=1, keepdim=True)[0]
                keep = (q_demo > q_mu).float()
        else:
            keep = torch.ones_like(err)
        return (err * keep).mean(), float(keep.mean())

    def pretrain_bc(self, steps: int, batch_size: int = 256) -> float:
        """The actor alone, by behaviour cloning on the demonstrations, before any environment step: the
        deterministic policy starts as a copy of the script. Returns the mean loss of the last 100 steps."""
        self.policy.set_training_mode(True)
        losses = []
        for _ in range(steps):
            d = self.replay_buffer.sample_demos(batch_size, env=self._vec_normalize_env)
            bc, _ = self.bc_term(d.observations, self.replay_buffer.last_clean)
            self.actor.optimizer.zero_grad()
            bc.backward()
            self.actor.optimizer.step()
            losses.append(bc.item())
        return float(np.mean(losses[-100:])) if losses else float("nan")

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """SB3 2.x SAC.train, plus the behaviour cloning term in the actor loss."""
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coefs, actor_losses, critic_losses = [], [], []

        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            n_demo = getattr(self.replay_buffer, "last_n_demo", 0)
            discounts = data.discounts if getattr(data, "discounts", None) is not None else self.gamma

            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            log_prob = log_prob.reshape(-1, 1)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q = torch.min(next_q, dim=1, keepdim=True)[0] - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = data.rewards + (1 - data.dones) * discounts * next_q

            current_q = self.critic(data.observations, data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_pi = torch.min(torch.cat(self.critic(data.observations, actions_pi), dim=1), dim=1, keepdim=True)[0]
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            if self.det_weight > 0:
                mu = self.actor(data.observations, deterministic=True)
                q_mu = torch.min(torch.cat(self.critic(data.observations, mu), dim=1), dim=1, keepdim=True)[0]
                actor_loss = actor_loss - self.det_weight * q_mu.mean()
            if self.bc_weight > 0 and n_demo > 0:
                bc, kept = self.bc_term(data.observations[-n_demo:], self.replay_buffer.last_clean, self.bc_grip)
                actor_loss = actor_loss / q_pi.abs().mean().detach().clamp(min=1.0) + self.bc_weight * bc
                self.bc_stats["bc"].append(bc.item())
                self.bc_stats["kept"].append(kept)
            actor_losses.append(actor_loss.item())
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def pop_bc_stats(self) -> dict:
        """Mean BC loss and share kept by the Q filter since the last call."""
        out = {k: round(float(np.mean(v)), 3) if v else None for k, v in self.bc_stats.items()}
        self.bc_stats = {"bc": [], "kept": []}
        return out
