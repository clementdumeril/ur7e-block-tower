import numpy as np
import pytest

from pensig.config import TowerConfig
from pensig.residual import ResidualTowerEnv
from pensig.tower import BLOCKS, TowerEnv
from pensig.tower_scripted import ScriptedStack


@pytest.fixture(scope="module")
def env():
    return TowerEnv(TowerConfig().noiseless())


def test_blocks_rest_on_table(env):
    for seed in range(10):
        env.reset(seed=seed)
        before = {b: env.block_pose(b)[0] for b in BLOCKS}
        for _ in range(20):
            env.step(np.zeros(5))
        for b in BLOCKS:
            assert np.linalg.norm(env.block_pose(b)[0] - before[b]) < 0.003, (env.stage, b)
            assert env.upright(b)


def test_step_api(env):
    for stage in (0, 1):
        obs, _ = env.reset(seed=1, options={"stage": stage})
        assert obs.shape == env.observation_space.shape
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        assert {"is_success", "dropped", "stopped", "stage"} <= info.keys()


def test_residual_obs_shape():
    env = ResidualTowerEnv(TowerConfig())
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    obs, *_ = env.step(np.zeros(5))
    assert obs.shape == env.observation_space.shape


def test_scripted_tower_without_noise(env):
    policy = ScriptedStack()
    towers = 0
    for seed in range(10):
        env.reset(seed=100 + seed, options={"stage": 0})
        built = 0
        for stage in (0, 1):
            if stage == 1:
                env.next_stage()
            policy.reset()
            done = False
            while not done:
                _, _, term, trunc, info = env.step(policy(env))
                done = term or trunc
            if not info["is_success"]:
                break
            built += 1
        towers += built == 2
    assert towers >= 8


def test_wrist_estimate_before_descent():
    env = TowerEnv(TowerConfig())
    policy = ScriptedStack()
    for seed in range(3):
        env.reset(seed=seed, options={"stage": 0})
        policy.reset()
        assert not env.wrist_held
        while policy.phase < 1:
            env.step(policy(env))
        assert env.wrist_held
        assert np.linalg.norm(env.est_held[:2] - env.block_pose(env.held)[0][:2]) < 0.005


def test_script_start_hands_over_at_phase():
    from pensig.curriculum import ScriptStart, frontier

    assert [frontier(k) for k in (0, 7, 8, 9)] == [7, 0, -1, -1]
    w = ScriptStart(TowerEnv(TowerConfig()), seed=3)
    w.set_level(3)
    obs, info = w.reset(seed=3)
    assert obs.shape == (29,) and w.unwrapped.t == 0
    assert 4 <= info["start_phase"] <= 7
    _, _, _, _, info = w.step(np.zeros(5))
    assert "start_phase" in info


def test_demo_buffer_mixes_batches_and_critic_has_layernorm():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import SAC
    from torch import nn

    from pensig.curriculum import ScriptStart
    from pensig.rlpd import DemoMixBuffer, LNSACPolicy

    w = ScriptStart(TowerEnv(TowerConfig()), seed=5)
    demo, wins = w.script_episodes(1)
    n = len(demo["obs"])
    assert n > 10 and demo["done"][-1] == 1 and set(np.unique(demo["act"][:, 4])) <= {-1.0, 1.0}
    model = SAC(LNSACPolicy, w, buffer_size=1000, batch_size=8, learning_starts=0,
                replay_buffer_class=DemoMixBuffer, replay_buffer_kwargs=dict(demo_ratio=0.5), device="cpu")
    assert any(isinstance(m, nn.LayerNorm) for m in model.critic.modules())
    model.replay_buffer.set_demos(demo["obs"], demo["act"], demo["rew"], demo["next"], demo["done"])
    model.learn(16)
    batch = model.replay_buffer.sample(8)
    assert batch.observations.shape == (8, 29) and batch.rewards.shape == (8, 1)
