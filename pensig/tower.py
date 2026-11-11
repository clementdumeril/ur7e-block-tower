"""Stack the lab blocks into a tower with a UR arm and a Hand-E gripper.

One episode places one block: the middle block on the base (stage 0) or the top block on the middle one
(stage 1). next_stage() chains the two without resetting the physics, as on the real arm.
Same action as the pen grasp: TCP displacement (x, y, z), yaw increment, gripper (-1 close, +1 open).
Block poses come from a camera estimate drawn once per stage, replaced by a closer wrist-camera estimate once the
TCP is above the block; the wrist force (the UR e-Series sensor, zeroed at the start of the stage) tells when the
block touches down.
"""

import gymnasium as gym
import mujoco
import numpy as np

from pensig.config import TowerConfig
from pensig.env import ARM_ACTUATORS, GRIP_SPEED, quat_z
from pensig.ik import Q_SEED, DiffIK, down_rotation
from pensig.scene import FINGER_LEN, HANDE_BODY, MAX_OPEN, TCP_Z, build_model

BLOCKS = ("green", "black", "red")  # bottom to top, as in the lab
STAGES = (("black", "green"), ("red", "black"))  # (held, target) per stage
BLOCK_RGBA = {"green": "0.2 0.75 0.3 1", "black": "0.06 0.06 0.06 1", "red": "0.85 0.1 0.08 1"}
BLOCK_MAX = 0.03  # half size the boxes are compiled at, shrunk at reset
TIP_BELOW_TCP = HANDE_BODY + FINGER_LEN - TCP_Z
OBS_CENTER = np.array([0.5, 0.0, 0.1])
SHAPING_GAMMA = 0.98  # TowerConfig.shaping_gamma default: the PPO discount in scripts/train.py
SUCCESS_BONUS = 10.0
FAILURE_PENALTY = 3.0  # protective stop, dropped, toppled or knocked block
COS_UPRIGHT = np.cos(np.deg2rad(5.0))
COS_TOPPLED = np.cos(np.deg2rad(15.0))

WORLD_XML = f"""
<worldbody>
  <light name="sun" pos="0.5 0 1.6" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
  <geom name="table" type="plane" size="1.5 1.5 0.05" material="table" conaffinity="3"/>
  <geom name="pedestal" type="cylinder" size="0.08 0.1375" pos="0 0 0.1375" rgba="0.3 0.3 0.32 1"/>
  {"".join(f'''
  <body name="{b}" pos="0.5 {0.1 * k} {BLOCK_MAX}">
    <freejoint name="{b}_free"/>
    <geom name="{b}" type="box" size="{BLOCK_MAX} {BLOCK_MAX} {BLOCK_MAX}" mass="0.03" conaffinity="3"
          rgba="{BLOCK_RGBA[b]}" friction="0.6 0.005 0.0001" condim="4" solref="0.004 1"
          solimp="0.95 0.99 0.0005"/>
  </body>''' for k, b in enumerate(BLOCKS))}
</worldbody>
"""


def wrap_quarter(angle: float) -> float:
    """Angle difference for a square block held by a gripper: symmetric under a quarter turn."""
    return (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


def carry_latch(carried: bool, closed: bool, tcp, est_grasp, tol: float = 0.015) -> bool:
    """Whether the block was taken this stage, from what the real arm knows too: set when the gripper is closed
    with the TCP within tol of the grasp estimate on every axis (tower_ros.policy.closing_on_block), cleared when
    it opens there again (a grasp given up). Opening anywhere else, as over the tower, keeps it."""
    if np.all(np.abs(np.asarray(tcp, float) - est_grasp) < tol):
        return bool(closed)
    return carried


class TowerEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: TowerConfig | None = None, render_mode: str | None = None):
        self.cfg = cfg or TowerConfig()
        self.render_mode = render_mode
        self.model = build_model(WORLD_XML)
        self.model.opt.timestep = self.cfg.sim_dt
        self.model.opt.noslip_iterations = self.cfg.noslip_iterations
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(self.cfg.control_dt / self.cfg.sim_dt))
        self.ik = DiffIK(self.model)

        m = self.model
        self.finger_qadr = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_left", "finger_right")]
        # With the stiff coupling between the fingers, a closed gripper chatters in place and never reopens
        # (the pen grasp never reopens, so it did not show). More armature, as for a geared drive, fixes it;
        # stiffer limits keep the 25 mm stroke.
        for n in ("finger_left", "finger_right"):
            m.dof_armature[m.jnt_dofadr[m.joint(n).id]] = 0.1
            m.jnt_solref[m.joint(n).id] = (0.005, 1.0)
        self.qadr = {b: m.jnt_qposadr[m.joint(f"{b}_free").id] for b in BLOCKS}
        self.dadr = {b: m.jnt_dofadr[m.joint(f"{b}_free").id] for b in BLOCKS}
        self.arm_act = np.array([m.actuator(n).id for n in ARM_ACTUATORS])
        # Menagerie's servo gains lag the target by centimeters; the real arm tracks servoL closely.
        for i in self.arm_act:
            kp, kv = (10000.0, 300.0) if m.actuator_forcerange[i, 1] > 100 else (2000.0, 60.0)
            m.actuator_gainprm[i, 0] = kp
            m.actuator_biasprm[i, 1:3] = (-kp, -kv)
        self.grip_act = m.actuator("gripper").id
        self.tcp_sid = m.site("tcp").id
        self.ft_adr = m.sensor_adr[m.sensor("wrist_force").id]
        self.gid = {n: m.geom(n).id for n in ("pad_left", "pad_right", "hande_body", "camera_mount", "table", *BLOCKS)}
        self.bid = {b: m.body(b).id for b in BLOCKS}
        self.gripper_geoms = np.array([self.gid[n] for n in ("hande_body", "camera_mount", "pad_left", "pad_right")])
        self.pad_geoms = np.array([self.gid["pad_left"], self.gid["pad_right"]])

        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), np.float32)
        self._renderer = None
        self._camera = None

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        stage = (options or {}).get("stage")
        for _ in range(20):
            if self._randomize(self.np_random, stage) and self._settle():
                break
        else:
            raise RuntimeError("could not place stable blocks in 20 tries")
        self._start_stage(self.stage)
        return self._obs(), self._info(False)

    def next_stage(self):
        """After the middle block is on the base: go on with the top block, from the current state."""
        if self.stage != 0:
            raise RuntimeError("next_stage() follows stage 0")
        self._start_stage(1)
        return self._obs(), self._info(False)

    def _randomize(self, rng: np.random.Generator, stage: int | None) -> bool:
        cfg, m, d = self.cfg, self.model, self.data
        u = lambda r: rng.uniform(*r)  # noqa: E731
        if stage is None:
            p = np.asarray(cfg.stage_probs, float)
            stage = int(rng.choice(len(STAGES), p=p / p.sum()))

        mujoco.mj_resetData(m, d)
        self.sizes = {}
        for b, (w, h) in zip(BLOCKS, (cfg.base_size, cfg.mid_size, cfg.top_size)):
            w += rng.uniform(-cfg.size_jitter, cfg.size_jitter)
            h += rng.uniform(-cfg.size_jitter, cfg.size_jitter)
            if b != "green":
                w = min(w, 2 * MAX_OPEN - 0.006)
            half = (w / 2, w / 2, h / 2)
            g, bid = self.gid[b], self.bid[b]
            m.geom_size[g] = half
            m.geom_aabb[g] = (0, 0, 0, *half)
            m.geom_rbound[g] = np.linalg.norm(half)
            m.geom_friction[g, 0] = u(cfg.block_friction)
            mass = u(cfg.block_mass)
            m.body_mass[bid] = mass
            m.body_inertia[bid] = mass / 12 * np.array([w**2 + h**2, w**2 + h**2, 2 * w**2])
            m.body_ipos[bid] = 0
            m.body_iquat[bid] = (1, 0, 0, 0)
            self.sizes[b] = np.array([w, h])
        for pad in ("pad_left", "pad_right"):
            m.geom_friction[self.gid[pad], 0] = u(cfg.pad_friction)
        force = u(cfg.grip_force)
        m.actuator_forcerange[self.grip_act] = (-force, force)
        # The contact softness MuJoCo derives from the masses (body_invweight0 and co) is computed at compile
        # time, for 30 g blocks. Left there, a 50-60 g block held off-centre made the solver diverge: 22 of 100
        # noisy grasps, most of them on heavy blocks. Recompute it for the new masses and sizes.
        mujoco.mj_setConst(m, d)

        on_table = ["green", "black", "red"] if stage == 0 else ["green", "red"]
        spots = []
        for _ in range(200):
            p = rng.uniform(cfg.xy_lo, cfg.xy_hi)
            if all(np.linalg.norm(p - q) >= cfg.min_gap for q in spots):
                spots.append(p)
                if len(spots) == len(on_table):
                    break
        else:
            return False
        poses = {b: (np.array([*p, self.sizes[b][1] / 2]), rng.uniform(-np.pi, np.pi))
                 for b, p in zip(on_table, spots)}
        if stage == 1:
            r, a = cfg.stack_offset * np.sqrt(rng.random()), rng.uniform(-np.pi, np.pi)
            base_pos, base_yaw = poses["green"]
            poses["black"] = (np.array([base_pos[0] + r * np.cos(a), base_pos[1] + r * np.sin(a),
                                      self.sizes["green"][1] + self.sizes["black"][1] / 2]),
                            base_yaw + rng.normal(0, cfg.stack_yaw))
        for b, (pos, yaw) in poses.items():
            qa = self.qadr[b]
            d.qpos[qa:qa + 3] = pos
            d.qpos[qa + 3:qa + 7] = quat_z(yaw)

        held = STAGES[stage][0]
        tallest = max(pos[2] + self.sizes[b][1] / 2 for b, (pos, _) in poses.items())
        start = np.array([*(poses[held][0][:2] + rng.uniform(-cfg.start_xy_spread, cfg.start_xy_spread, 2)),
                          tallest + TIP_BELOW_TCP + u(cfg.start_above)])
        start = np.clip(start, cfg.target_lo, cfg.target_hi)
        yaw = rng.uniform(-np.pi, np.pi)
        q = self.ik.solve(Q_SEED, start, down_rotation(yaw), iters=200, tol=1e-7)
        if np.linalg.norm(self.ik.fk(q)[0] - start) > 2e-3:
            return False
        d.qpos[self.ik.qadr] = q
        d.ctrl[self.arm_act] = q
        d.qpos[self.finger_qadr] = MAX_OPEN
        d.ctrl[self.grip_act] = GRIP_SPEED
        mujoco.mj_forward(m, d)

        self.q_cmd = q
        self.target_pos = start
        self.target_yaw = yaw
        self.stage = stage
        return True

    def _settle(self) -> bool:
        d = self.data
        before = {b: d.xpos[self.bid[b]].copy() for b in BLOCKS}
        for _ in range(150):
            mujoco.mj_step(self.model, d)
        return all(np.linalg.norm(d.xpos[self.bid[b]] - before[b]) < 0.003 and self.upright(b) for b in BLOCKS)

    def _start_stage(self, stage: int):
        cfg, rng, m = self.cfg, self.np_random, self.model
        self.stage = stage
        self.held, self.target = STAGES[stage]
        others = [self.gid[b] for b in BLOCKS if b != self.held]
        self._is_pressing = np.zeros(m.ngeom, bool)
        self._is_pressing[[*self.gripper_geoms, self.gid[self.held]]] = True
        self._is_obstacle = np.zeros(m.ngeom, bool)
        self._is_obstacle[[self.gid["table"], *others]] = True

        hp, hyaw = self.block_pose(self.held)
        tp, tyaw = self.block_pose(self.target)
        self.est_held = hp + self._noise(cfg.pos_noise, 3)
        self.est_held_yaw = hyaw + self._noise(cfg.yaw_noise)
        self.est_size = self.sizes[self.held] + self._noise(cfg.size_noise, 2)
        self.est_top = tp + [0, 0, self.sizes[self.target][1] / 2] + self._noise(cfg.pos_noise, 3)
        self.est_target_yaw = tyaw + self._noise(cfg.yaw_noise)
        self.est_target_width = self.sizes[self.target][0] + self._noise(cfg.size_noise)
        # The TCP holds the block this far above its bottom, so it ends this far above the target's top.
        self.grasp_z_off = TIP_BELOW_TCP + cfg.finger_above_bottom
        self.wrist_held = self.wrist_target = False
        self.carried = False
        self._update_targets()

        self.start_pos = {b: self.data.xpos[self.bid[b]].copy() for b in BLOCKS if b != self.held}
        self.bottom0 = hp[2] - self.sizes[self.held][1] / 2
        self.lifted = False
        self.phi_prev = None
        self.t = 0
        self.hold = 0
        self.prev_action = np.zeros(5)
        self.queued = np.zeros(5)
        self.latency = rng.random() < cfg.latency_prob
        self.ft_tare = self.data.sensordata[self.ft_adr:self.ft_adr + 3].copy()
        self._read_force()
        self._wrist_look()

    def _noise(self, sigma: float, n: int | None = None):
        if not sigma:
            return np.zeros(n) if n else 0.0
        return self.np_random.normal(0, sigma, n)

    def _update_targets(self):
        self.est_grasp = self.est_held + [0, 0, self.grasp_z_off - self.est_size[1] / 2]
        self.est_place = self.est_top + [0, 0, self.grasp_z_off]

    def _wrist_look(self):
        """Close-up from the wrist camera: once the TCP is above a block's first estimate, a better one.
        The held block only before it is lifted; in the hand it hides from the camera."""
        cfg = self.cfg
        if not cfg.wrist_camera:
            return
        tcp = self.data.site_xpos[self.tcp_sid]

        def above(est_top):
            h = tcp[2] - TIP_BELOW_TCP - est_top[2]
            return np.linalg.norm(tcp[:2] - est_top[:2]) < cfg.wrist_radius and -0.005 < h < cfg.wrist_max_height

        if not self.wrist_held and not self.lifted and above(self.est_held + [0, 0, self.est_size[1] / 2]):
            hp, hyaw = self.block_pose(self.held)
            self.est_held = hp + self._noise(cfg.wrist_pos_noise, 3)
            self.est_held_yaw = hyaw + self._noise(cfg.wrist_yaw_noise)
            self.wrist_held = True
        if not self.wrist_target and above(self.est_top):
            tp, tyaw = self.block_pose(self.target)
            self.est_top = tp + [0, 0, self.sizes[self.target][1] / 2] + self._noise(cfg.wrist_pos_noise, 3)
            self.est_target_yaw = tyaw + self._noise(cfg.wrist_yaw_noise)
            self.wrist_target = True
        self._update_targets()

    # ------------------------------------------------------------------- step

    def step(self, action):
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        applied = self.queued if self.latency else a
        self.queued = a

        self.target_pos = np.clip(self.target_pos + applied[:3] * cfg.max_dpos, cfg.target_lo, cfg.target_hi)
        self.target_yaw = (self.target_yaw + applied[3] * cfg.max_dyaw + np.pi) % (2 * np.pi) - np.pi
        stopped = self.servo(self.target_pos, down_rotation(self.target_yaw), applied[4])
        self.t += 1
        self._read_force()

        if self._unstable() or self._blown_up():
            # MuJoCo has already reset the state; end the episode without a learning signal.
            info = self._info(False)
            info["unstable"] = True
            return self._obs(), 0.0, False, True, info

        reward, success, failure = self._reward(a)
        self._wrist_look()
        self.prev_action = a
        self.carried = carry_latch(self.carried, a[4] < 0, self.data.site_xpos[self.tcp_sid], self.est_grasp)
        if stopped:
            reward -= FAILURE_PENALTY
            success = False
        terminated = success or bool(failure) or stopped
        truncated = self.t >= cfg.episode_steps
        if truncated and not terminated:
            reward -= cfg.timeout_penalty
        info = self._info(success, failure)
        info["stopped"] = stopped
        return self._obs(), reward, terminated, truncated, info

    def servo(self, pos: np.ndarray, rot: np.ndarray, grip: float) -> bool:
        """One control period towards a TCP pose, like servoL. Returns True on a protective stop."""
        m, d = self.model, self.data
        q_new = self.ik.solve(self.q_cmd, pos, rot)
        d.ctrl[self.grip_act] = grip * GRIP_SPEED
        stopped = False
        for k in range(self.n_sub):
            d.ctrl[self.arm_act] = self.q_cmd + (k + 1) / self.n_sub * (q_new - self.q_cmd)
            mujoco.mj_step(m, d)
            if self._unstable() or self._blown_up():
                break
            if self._protective_stop():
                stopped = True
                break
        self.q_cmd = q_new
        return stopped

    def _protective_stop(self) -> bool:
        """Like the UR protective stop: the gripper or the held block pressed hard against the table or
        another block, or a block knocked away. A released block falling onto the tower does not count: the arm
        only feels the block while a finger touches it. Nor does the held block's speed: the fingers closing or
        opening on it give 2-3 m/s for one 2 ms step, harmless, and the UR does not watch blocks."""
        d = self.data
        for b in BLOCKS:
            if b == self.held:
                continue
            v = d.qvel[self.dadr[b]:self.dadr[b] + 3]
            if v @ v > 4.0:
                return True
        n = d.ncon
        if n == 0:
            return False
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        held = self.gid[self.held]
        pads = np.isin(g1, self.pad_geoms) & (g2 == held) | np.isin(g2, self.pad_geoms) & (g1 == held)
        self._is_pressing[held] = bool(pads.any())
        pressing = (self._is_pressing[g1] & self._is_obstacle[g2]) | (self._is_pressing[g2] & self._is_obstacle[g1])
        adr = d.contact.efc_address[:n][pressing]
        adr = adr[adr >= 0]
        # With elliptic cones the first constraint row of a contact is its normal force.
        return bool(adr.size and d.efc_force[adr].max() > self.cfg.stop_force)

    def _blown_up(self) -> bool:
        """A block faster than 10 m/s: the solver diverged, nothing the arm did. Ends the episode as unstable,
        without the protective stop's penalty."""
        v = self.data.qvel
        return any(v[self.dadr[b]:self.dadr[b] + 3] @ v[self.dadr[b]:self.dadr[b] + 3] > 100.0 for b in BLOCKS)

    def _unstable(self) -> bool:
        w = self.data.warning
        return any(w[i].number > 0 for i in (mujoco.mjtWarning.mjWARN_BADQACC,
                                             mujoco.mjtWarning.mjWARN_BADQVEL,
                                             mujoco.mjtWarning.mjWARN_BADQPOS))

    def _read_force(self):
        f = self.data.sensordata[self.ft_adr:self.ft_adr + 3] - self.ft_tare
        if self.cfg.force_noise:
            f = f + self.np_random.normal(0, self.cfg.force_noise, 3)
        self.force = f

    def block_pose(self, b: str) -> tuple[np.ndarray, float]:
        x = self.data.xmat[self.bid[b]].reshape(3, 3)[:, 0]
        return self.data.xpos[self.bid[b]].copy(), float(np.arctan2(x[1], x[0]))

    def upright(self, b: str, cos_max: float = COS_UPRIGHT) -> bool:
        return bool(self.data.xmat[self.bid[b]][8] > cos_max)

    def _reward(self, a: np.ndarray) -> tuple[float, bool, str]:
        cfg, d = self.cfg, self.data
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]

        def touching(x, y):
            return bool(np.any(((g1 == x) & (g2 == y)) | ((g1 == y) & (g2 == x))))

        held, target = self.gid[self.held], self.gid[self.target]
        left = touching(held, self.gid["pad_left"])
        right = touching(held, self.gid["pad_right"])
        grasped = left and right
        hit = bool(np.any((np.isin(g1, self.gripper_geoms) & self._is_obstacle[g2])
                          | (np.isin(g2, self.gripper_geoms) & self._is_obstacle[g1])))

        tcp = d.site_xpos[self.tcp_sid]
        hp, hyaw = self.block_pose(self.held)
        tp, tyaw = self.block_pose(self.target)
        h_held = self.sizes[self.held][1]
        bottom = hp - [0, 0, h_held / 2]
        top = tp + [0, 0, self.sizes[self.target][1] / 2]
        dist = np.linalg.norm(tcp - (bottom + [0, 0, self.grasp_z_off]))
        d_place = np.linalg.norm(bottom - top)
        self.place_err = float(np.linalg.norm(hp[:2] - tp[:2]))
        self.yaw_err = abs(wrap_quarter(hyaw - tyaw))
        on_target = touching(held, target) and abs(bottom[2] - top[2]) < 0.003 and self.upright(self.held)
        placed = on_target and self.place_err < cfg.place_tol and self.yaw_err < cfg.place_yaw_tol
        released = placed and not (left or right)
        clear = tcp[2] - TIP_BELOW_TCP > hp[2] + h_held / 2 + cfg.clear_height
        if grasped and bottom[2] > self.bottom0 + 0.01:
            self.lifted = True

        # Progress: reach, grasp, carry, then a released placement above anything reachable while holding.
        if released:
            phi = 3.0 + 0.5 * (1.0 - self.place_err / cfg.place_tol)
        else:
            phi = 1.0 - np.tanh(10.0 * dist) + 0.5 * grasped + 1.0 * (1.0 - np.tanh(10.0 * d_place)) * grasped
        # Paid as a potential difference, not per step. Paid per step, holding the block over the tower
        # earned 2.5 every step while the success bonus came 20 steps later, discounted by gamma: with half the
        # placements succeeding, PPO learned to stall there.
        if self.phi_prev is None:
            self.phi_prev = phi
        reward = cfg.shaping_gamma * phi - self.phi_prev
        self.phi_prev = phi
        reward -= 0.2 * hit
        reward -= 0.01 * float(a @ a)

        self.hold = self.hold + 1 if (released and clear) else 0
        success = self.hold >= cfg.hold_steps
        failure = ""
        if any(not self.upright(b, COS_TOPPLED) for b in BLOCKS):
            failure = "toppled"
        elif any(np.linalg.norm(d.xpos[self.bid[b]] - p) > 0.01 for b, p in self.start_pos.items()):
            failure = "knocked"
        elif self.lifted and not grasped and touching(held, self.gid["table"]):
            failure = "dropped"
        if success:
            reward += SUCCESS_BONUS
        if failure:
            reward -= FAILURE_PENALTY
        return float(reward), bool(success), failure

    # ---------------------------------------------------------------- outputs

    def tcp_yaw(self) -> float:
        x = self.data.site_xmat[self.tcp_sid].reshape(3, 3)[:, 0]
        return float(np.arctan2(x[1], x[0]))

    def tower_height(self) -> float:
        """Height above the table of the highest block top."""
        return max(self.data.xpos[self.bid[b]][2] + self.sizes[b][1] / 2 for b in BLOCKS)

    def _obs(self) -> np.ndarray:
        d = self.data
        tcp = d.site_xpos[self.tcp_sid]
        yaw = self.tcp_yaw()
        opening = d.qpos[self.finger_qadr[0]] / MAX_OPEN
        obs = np.concatenate([
            tcp - OBS_CENTER,
            [np.cos(yaw), np.sin(yaw)],
            [opening],
            self.est_grasp - tcp,
            [wrap_quarter(self.est_held_yaw - yaw)],
            self.est_place - tcp,
            [wrap_quarter(self.est_target_yaw - yaw)],
            self.est_size * 10,
            [self.est_target_width * 10],
            [self.wrist_held, self.wrist_target],
            self.force / 10,
            self.prev_action,
            [self._slot27()],
            [self.stage],
        ])
        return obs.astype(np.float32)

    def _slot27(self) -> float:
        if self.cfg.carry_in_obs:
            return float(self.carried)
        return self.t / self.cfg.episode_steps if self.cfg.time_in_obs else 0.0

    def _info(self, success: bool, failure: str = "") -> dict:
        return {"is_success": success, "failure": failure, "dropped": failure == "dropped", "stopped": False,
                "stage": self.stage, "support": f"stage{self.stage}"}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._camera = mujoco.MjvCamera()
        cam = self._camera
        base = self.data.xpos[self.bid["green"]]
        cam.lookat[:] = (0.5 * (base[0] + 0.5), 0.5 * base[1], 0.07)
        cam.distance, cam.azimuth, cam.elevation = 0.75, 150, -20
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
