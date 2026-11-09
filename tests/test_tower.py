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


def test_stage_one_follows_the_policys_own_stage_zero():
    from pensig.curriculum import ScriptStart

    w = ScriptStart(TowerEnv(TowerConfig()), seed=4, chain=1.0)
    w.fixed_phase = 6  # the script opens and lifts: success within a few steps
    w.reset(seed=4, options={"stage": 0})
    e = w.unwrapped
    assert e.stage == 0
    done = False
    while not done:
        _, _, term, trunc, info = w.step(w.script(e))
        done = term or trunc
    assert info["is_success"]
    w.fixed_phase = None
    obs, info = w.reset()
    assert info.get("chained") and e.stage == 1 and e.t == 0 and info["start_phase"] == -1
    _, info = w.reset()
    assert not info.get("chained")  # only right after a stage 0 success


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


def test_demo_sac_clones_the_script_and_trains_with_the_bc_term():
    pytest.importorskip("stable_baselines3")
    import torch

    from pensig.curriculum import ScriptStart
    from pensig.rlpd import DemoMixBuffer, DemoSAC, LNSACPolicy

    torch.manual_seed(0)
    w = ScriptStart(TowerEnv(TowerConfig()), seed=5)
    demo, _ = w.script_episodes(2)
    model = DemoSAC(LNSACPolicy, w, buffer_size=1000, batch_size=32, learning_starts=0,
                    replay_buffer_class=DemoMixBuffer, replay_buffer_kwargs=dict(demo_ratio=0.5), device="cpu",
                    bc_weight=1.0, q_filter=True)
    assert np.array_equal(demo["clean"][:, 4], demo["act"][:, 4])  # the noise is only on the motion
    model.replay_buffer.set_demos(demo["obs"], demo["act"], demo["rew"], demo["next"], demo["done"], demo["clean"])
    obs, act = (torch.as_tensor(demo[k]) for k in ("obs", "clean"))
    before = float(model.bc_term(obs, act)[0])
    model.pretrain_bc(300, 64)
    after = float(model.bc_term(obs, act)[0])
    assert after < 0.5 * before
    model.learn(16)
    stats = model.pop_bc_stats()
    assert stats["bc"] is not None and 0.0 <= stats["kept"] <= 1.0
    assert model.replay_buffer.last_n_demo == 16
