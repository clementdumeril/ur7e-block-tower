"""The policy node's pure parts: observation, action mapping, frames, gripper, the exported network, and a
stage run in closed loop against an arm that tracks the command exactly. With MuJoCo installed, the
observation, the wrist trigger and the target update are also checked against TowerEnv itself."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_ros.geometry import BlockEstimate, ToolPose  # noqa: E402
from tower_ros.policy import (MAX_OPEN, OBS_CENTER, TIP_BELOW_TCP, GripperModel, NumpyPolicy,  # noqa: E402
                              PolicyFrame, SimConfig, StageState, closing_on_block, load_policy, residual_obs,
                              scripted_stack_class, sensor_force, sim_start, stack_below, stage_done, step_target,
                              total_action)

RUN = ROOT / "runs" / "tower_v6"
HEIGHTS = {"green": 0.05, "black": 0.08, "red": 0.03}
ScriptedStack = scripted_stack_class(ROOT)


def stage0_state(cfg=None, close_err=(0.002, -0.001)):
    cfg = cfg or SimConfig()
    held = (np.array([0.52, 0.10, 0.04]), 0.3)  # black block center, 80 mm tall
    target = (np.array([0.45, -0.10, 0.05]), -0.2)  # green top
    dx, dy = close_err
    held_close = (held[0] + [dx, dy, 0], held[1] + 0.02)
    target_close = (target[0] + [dy, dx, 0], target[1] - 0.02)
    tcp = sim_start(held[0], 0.08, 0.08, cfg)
    return StageState(cfg, 0, held, target, (0.04, 0.08), 0.05, tcp, 0.1, held_close, target_close)


# ---------------------------------------------------------------- observation

def test_obs_layout():
    s = stage0_state()
    s.force = np.array([1.0, -2.0, 30.0])
    s.prev_action = np.array([0.1, 0.2, 0.3, 0.4, -1.0])
    s.t = 40
    tcp, yaw = np.array([0.5, 0.1, 0.2]), 0.7
    o = s.obs(tcp, yaw, 0.8)
    assert o.shape == (29,) and o.dtype == np.float32
    np.testing.assert_allclose(o[0:3], tcp - OBS_CENTER, atol=1e-6)
    np.testing.assert_allclose(o[3:5], [math.cos(yaw), math.sin(yaw)], atol=1e-6)
    assert o[5] == pytest.approx(0.8)
    grasp = s.est_held + [0, 0, TIP_BELOW_TCP + 0.015 - 0.04]
    np.testing.assert_allclose(o[6:9], grasp - tcp, atol=1e-6)
    assert o[9] == pytest.approx(0.3 - 0.7, abs=1e-6)
    assert s.obs(tcp, 1.2, 0.8)[9] == pytest.approx(0.3 - 1.2 + math.pi / 2, abs=1e-6)  # modulo a quarter turn
    np.testing.assert_allclose(o[10:13], s.est_top + [0, 0, TIP_BELOW_TCP + 0.015] - tcp, atol=1e-6)
    np.testing.assert_allclose(o[14:17], [0.4, 0.8, 0.5], atol=1e-6)
    np.testing.assert_allclose(o[17:19], [0, 0])
    np.testing.assert_allclose(o[19:22], [0.1, -0.2, 3.0], atol=1e-6)
    np.testing.assert_allclose(o[22:27], s.prev_action, atol=1e-6)
    assert o[27] == pytest.approx(0.25) and o[28] == 0


def test_residual_obs_and_action():
    base = np.array([1.0, -0.5, 0.0, 0.2, -1.0])
    o = residual_obs(np.zeros(29), base, 5)
    assert o.shape == (42,)
    np.testing.assert_allclose(o[29:34], base)
    assert o[34 + 5] == 1 and o[34:].sum() == 1
    assert residual_obs(np.zeros(29), base, 12)[-1] == 1  # phase index clipped as in ResidualTowerEnv
    a = total_action(base, [1.0, 1.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(a, [1.0, 0.0, -0.5, 0.2, -0.7])


def test_wrist_trigger_swaps_in_the_closeup():
    s = stage0_state()
    s.wrist_look(s.est_held + [0.0, 0.05, 0.10])  # 5 cm to the side
    assert not s.wrist_held
    s.wrist_look(s.est_held + [0.005, 0.0, 0.04 + TIP_BELOW_TCP + 0.26])  # above, too high
    assert not s.wrist_held
    near = s.est_held + [0.005, 0, 0.04 + TIP_BELOW_TCP + 0.10]
    s.wrist_look(near)
    assert s.wrist_held and not s.wrist_target
    np.testing.assert_allclose(s.est_held[:2], [0.522, 0.099])
    assert s.est_held_yaw == pytest.approx(0.32)
    np.testing.assert_allclose(s.est_grasp, s.est_held + [0, 0, TIP_BELOW_TCP + 0.015 - 0.04])
    s.wrist_look(s.est_top + [0.012, 0, 0.1])  # 12 mm off: no
    assert not s.wrist_target
    s.wrist_look(s.est_top + [0.0, 0.009, 0.1])
    assert s.wrist_target and s.est_target_yaw == pytest.approx(-0.22)


def test_no_held_closeup_after_lift():
    s = stage0_state()
    s.lifted = True
    s.wrist_look(s.est_held + [0, 0, 0.1])
    assert not s.wrist_held


# ---------------------------------------------------------------- action and frames

def test_step_target_matches_the_sim_and_caps_speed():
    cfg = SimConfig()
    pos, yaw = step_target([0.5, 0.0, 0.2], 3.1, [1, -1, 0.5, 1, 0], cfg)
    np.testing.assert_allclose(pos, [0.515, -0.015, 0.2075])
    assert yaw == pytest.approx(3.25 - 2 * math.pi)
    pos, yaw = step_target([0.5, 0.0, 0.2], 0.0, [1, 1, 0, -1, 0], cfg, max_step=0.005, max_yaw_step=0.03)
    assert np.linalg.norm(pos - [0.5, 0.0, 0.2]) == pytest.approx(0.005)
    assert yaw == pytest.approx(-0.03)
    pos, _ = step_target([0.75, 0.0, 0.021], 0.0, [1, 0, -1, 0, 0], cfg)
    np.testing.assert_allclose(pos, [0.75, 0.0, 0.02])  # sim workspace clip


def test_frame_round_trip():
    fr = PolicyFrame(table_z=-0.206, yaw=0.3, offset=(0.1, -0.05), finger_axis_yaw=0.785)
    tool = ToolPose(0.4, 0.3, -0.1, 0.2)
    tcp, yaw = fr.tcp_from_tool(tool)
    back = fr.tool_from_tcp(tcp, yaw)
    np.testing.assert_allclose(back.xyz(), tool.xyz(), atol=1e-12)
    assert back.yaw == pytest.approx(tool.yaw)
    assert yaw == pytest.approx(0.2 + 0.785 + 0.3)
    flat = PolicyFrame(table_z=-0.206)
    tcp, _ = flat.tcp_from_tool(ToolPose(0.5, 0.0, -0.206 + 0.015, 0.0))
    np.testing.assert_allclose(tcp, [0.5, 0.0, 0.015 + TIP_BELOW_TCP])  # table at z = 0


def test_block_heights_from_the_table_ring():
    fr = PolicyFrame(table_z=-0.206)
    e = BlockEstimate(0.45, -0.1, top=-0.08, yaw=0.1, height=0.122)  # bottom -0.202, camera top 4 mm low
    p, yaw = fr.block(e, stack_below("black", HEIGHTS), HEIGHTS["black"])
    np.testing.assert_allclose(p, [0.45, -0.1, 0.004 + 0.05 + 0.04])
    assert stack_below("green", HEIGHTS) == 0 and stack_below("red", HEIGHTS) == pytest.approx(0.13)


def test_sensor_force_axes():
    fr = PolicyFrame(table_z=-0.2)
    # The tower pushes the held block (and the tool) up: the sim's sensor reads +z (its z points down).
    np.testing.assert_allclose(sensor_force([0, 0, 4.0], 0.0, fr, sign=-1.0), [0, 0, 4.0], atol=1e-12)
    # Along the closing axis at yaw 90 deg is base y.
    np.testing.assert_allclose(sensor_force([0, 1.0, 0], math.pi / 2, fr, sign=1.0), [1.0, 0, 0], atol=1e-12)


def test_sim_start():
    s = sim_start([0.5, 0.1, 0.04], 0.08, 0.08, SimConfig())
    np.testing.assert_allclose(s, [0.5, 0.1, 0.08 + TIP_BELOW_TCP + 0.12])


# ---------------------------------------------------------------- gripper

def test_gripper_hysteresis_and_opening():
    g = GripperModel(threshold=0.5)
    assert g.opening == 1.0
    assert g.wanted(-0.4) is None and g.wanted(0.9) is None
    assert g.wanted(-0.7) is True
    g.set(True, 0.04)
    assert g.opening == pytest.approx(0.02 / MAX_OPEN)
    assert g.wanted(-1.0) is None and g.wanted(0.3) is None
    assert g.wanted(0.7) is False
    g.set(False, 0.04)
    assert g.opening == 1.0 and g.width == 0.0
    g.set(True, 0.0)
    assert g.opening == 0.0
    assert closing_on_block([0.5, 0.1, 0.03], np.array([0.505, 0.095, 0.035]))
    assert not closing_on_block([0.5, 0.1, 0.06], np.array([0.5, 0.1, 0.035]))


# ---------------------------------------------------------------- closed loop

def run_stage(policy=None, steps=160):
    """One stage against an arm that tracks the command exactly; the close-ups are the true poses, and the
    wrist force rises once the held block meets the target."""
    s = stage0_state()
    true_place = s.target_close[0][2] + s.grasp_z_off
    script, g = ScriptedStack(), GripperModel()
    tcp, yaw = s.target_pos.copy(), s.target_yaw
    s.wrist_look(tcp)
    base = script(s)
    phases = []
    close_z = None
    while s.t < steps:
        obs = residual_obs(s.obs(tcp, yaw, g.opening), base, script.phase)
        assert obs.shape == (42,) and np.isfinite(obs).all()
        res = np.zeros(5) if policy is None else policy(obs)
        a = total_action(base, res)
        s.target_pos, s.target_yaw = step_target(s.target_pos, s.target_yaw, a, s.cfg)
        want = g.wanted(a[4])
        if want is not None:
            g.set(want, 0.04 if want and closing_on_block(tcp, s.est_grasp) else 0.0)
            close_z = tcp[2] if want else None
        tcp, yaw = s.target_pos.copy(), s.target_yaw
        s.t += 1
        pressing = g.width and tcp[2] < true_place
        s.force = np.array([0.0, 0.0, 4.0 if pressing else 0.0])
        if g.width and close_z is not None and tcp[2] > close_z + 0.01:
            s.lifted = True
        s.wrist_look(tcp)
        s.prev_action = a
        base = script(s)
        phases.append(script.phase)
        if stage_done(script, s):
            return s, phases, g
    return s, phases, g


def test_scripted_stage_closes_the_loop():
    s, phases, g = run_stage()
    assert stage_done(ScriptedStack(), s) is False  # a fresh script is not done
    assert phases[-1] == 7 and sorted(set(phases)) == list(range(8))
    assert s.wrist_held and s.wrist_target and s.lifted
    assert s.t < 160 and not g.closed
    assert np.linalg.norm(s.target_pos[:2] - s.target_close[0][:2]) < 0.003  # went down over the close-up


# ---------------------------------------------------------------- the trained network

@pytest.mark.skipif(not (RUN / "policy.npz").exists(), reason="runs/tower_v6/policy.npz not exported here")
def test_policy_closes_the_loop():
    s, phases, _ = run_stage(NumpyPolicy(RUN / "policy.npz"))
    assert phases[-1] == 7 and s.t < 160


def test_numpy_policy_matches_sb3(tmp_path):
    pytest.importorskip("stable_baselines3")
    if not (RUN / "final_model.zip").exists():
        pytest.skip("runs/tower_v6 not here")
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_tower_policy import export

    export(RUN, tmp_path / "policy.npz")
    mine, sb3 = NumpyPolicy(tmp_path / "policy.npz"), load_policy(RUN / "final_model.zip")
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = (mine.mean + rng.normal(0, 2, 42) * np.sqrt(mine.var)).astype(np.float32)
        np.testing.assert_allclose(mine(obs), sb3(obs), atol=1e-5)


# ---------------------------------------------------------------- against the sim itself

def test_constants_match_pensig():
    pytest.importorskip("mujoco")
    import pensig.tower as tower
    from pensig.config import TowerConfig
    from pensig.residual import ResidualTowerEnv
    from pensig.scene import MAX_OPEN as SIM_MAX_OPEN

    assert TIP_BELOW_TCP == pytest.approx(tower.TIP_BELOW_TCP)
    assert MAX_OPEN == SIM_MAX_OPEN
    np.testing.assert_allclose(OBS_CENTER, tower.OBS_CENTER)
    cfg, sim = SimConfig(), TowerConfig()
    for k in ("control_dt", "episode_steps", "max_dpos", "max_dyaw", "target_lo", "target_hi",
              "finger_above_bottom", "wrist_radius", "wrist_max_height"):
        assert getattr(cfg, k) == getattr(sim, k), k
    env = ResidualTowerEnv(sim)
    np.testing.assert_allclose(env.scale, [0.5, 0.5, 0.5, 0.5, 0.3])
    assert env.n_phases == 8 and env.observation_space.shape == (42,)


def test_obs_and_trigger_match_tower_env():
    """Noiseless TowerEnv driven by the script: a StageState fed with the env's TCP, force and lift flag
    gives the same 29 values, the same wrist flags and the same commanded target at every step."""
    pytest.importorskip("mujoco")
    from pensig.config import TowerConfig
    from pensig.tower import TowerEnv

    sim = TowerConfig().noiseless()
    sim.mid_size, sim.top_size = (0.04, 0.08), (0.035, 0.03)
    env = TowerEnv(sim)
    env.reset(seed=4, options={"stage": 0})
    cfg = SimConfig(**{k: getattr(sim, k) for k in SimConfig.__dataclass_fields__})
    hp, hyaw = env.block_pose(env.held)
    tp, tyaw = env.block_pose(env.target)
    top = tp + [0, 0, env.sizes[env.target][1] / 2]
    tcp = env.data.site_xpos[env.tcp_sid].copy()
    s = StageState(cfg, 0, (env.est_held, env.est_held_yaw), (env.est_top, env.est_target_yaw), env.est_size,
                   env.est_target_width, env.target_pos, env.target_yaw, (hp, hyaw), (top, tyaw))
    s.force = env.force.copy()
    s.wrist_look(tcp)
    sim_script, my_script = ScriptedStack(), ScriptedStack()
    for _ in range(120):
        opening = env.data.qpos[env.finger_qadr[0]] / MAX_OPEN
        np.testing.assert_allclose(s.obs(tcp, env.tcp_yaw(), opening), env._obs(), atol=1e-5)
        a = sim_script(env)
        np.testing.assert_allclose(my_script(s), a, atol=1e-7)  # float rounding in the estimates: ~1e-9 at times
        env.step(a)
        s.target_pos, s.target_yaw = step_target(s.target_pos, s.target_yaw, a, cfg)
        np.testing.assert_allclose(s.target_pos, env.target_pos, atol=1e-12)
        assert s.target_yaw == pytest.approx(env.target_yaw)
        tcp = env.data.site_xpos[env.tcp_sid].copy()
        s.t, s.force, s.lifted, s.prev_action = env.t, env.force.copy(), env.lifted, np.clip(a, -1, 1)
        s.wrist_look(tcp)
        assert (s.wrist_held, s.wrist_target) == (env.wrist_held, env.wrist_target)
        if sim_script.phase == 7:
            break
    assert sim_script.phase == 7


def test_grasp_rises_on_a_tall_block_for_new_runs():
    """TowerConfig.palm_above_top: on the 80 mm black block the grasp is 32 mm over its bottom, not 15 mm; runs
    trained before the field keep the bottom rule."""
    old = stage0_state()
    np.testing.assert_allclose(old.est_grasp, old.est_held + [0, 0, TIP_BELOW_TCP + 0.015 - 0.04])
    new = stage0_state(SimConfig(palm_above_top=0.012))
    np.testing.assert_allclose(new.est_grasp, new.est_held + [0, 0, TIP_BELOW_TCP + 0.032 - 0.04])
    np.testing.assert_allclose(new.est_place, new.est_top + [0, 0, TIP_BELOW_TCP + 0.032])
