"""The residual PPO tower policy (pensig/residual.py ResidualTowerEnv) on the real arm, free of ROS.

The sim's world frame has the table at z = 0 and the arm base at the origin; its TCP is a point 12 mm above
the fingertips and its yaw is the direction the fingers close along. PolicyFrame maps base_link and the
fingertip ToolPose of tower_ros.geometry to that frame, StageState holds what TowerEnv knows during a stage
(estimates, wrist close-up flags, commanded target, force) and is also the object ScriptedStack reads, so
the script logic is pensig's own.
"""

import json
import math
import sys
import types
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from tower_ros.geometry import BLOCKS, BlockEstimate, ToolPose, wrap_pi, wrap_quarter

# pensig/tower.py and pensig/scene.py, copied so this module imports without MuJoCo (the tests compare them).
TIP_BELOW_TCP = 0.012  # HANDE_BODY + FINGER_LEN - TCP_Z: the sim TCP is this far above the fingertips
FINGER_LEN = 0.06  # pensig/scene.py, palm to fingertips
MAX_OPEN = 0.025  # Hand-E stroke per finger
OBS_CENTER = np.array([0.5, 0.0, 0.1])
N_PHASES = 8  # ScriptedStack phases, one-hot in the residual observation
RESIDUAL_SCALE = np.array([0.5, 0.5, 0.5, 0.5, 0.3])  # ResidualTowerEnv scale


@dataclass
class SimConfig:
    """The TowerConfig fields the observation, the action and ScriptedStack read, as trained (config.json)."""
    control_dt: float = 0.05
    episode_steps: int = 160
    max_dpos: float = 0.015
    max_dyaw: float = 0.15
    target_lo: tuple = (0.25, -0.35, 0.02)
    target_hi: tuple = (0.75, 0.35, 0.45)
    finger_above_bottom: float = 0.015
    palm_above_top: float | None = None  # TowerConfig.palm_above_top; None (bottom rule only) for older runs
    wrist_camera: bool = True
    wrist_radius: float = 0.01
    wrist_max_height: float = 0.25
    start_above: tuple = (0.08, 0.16)
    xy_lo: tuple = (0.38, -0.18)
    xy_hi: tuple = (0.60, 0.18)
    time_in_obs: bool = True  # runs from before the field (tower_v6) had the clock in obs[27]
    carry_in_obs: bool = False  # TowerConfig.carry_in_obs, the carry latch in obs[27]; False for older runs

    @classmethod
    def from_run(cls, run_dir) -> "SimConfig":
        path = Path(run_dir) / "config.json"
        if not path.exists():
            return cls()
        env = json.loads(path.read_text())["env"]
        return cls(**{f.name: (tuple(env[f.name]) if isinstance(env[f.name], list) else env[f.name])
                      for f in fields(cls) if f.name in env})


def down_rotation(yaw: float) -> np.ndarray:
    """pensig.ik.down_rotation: the sim TCP frame, pointing down, x (the closing axis) at `yaw`."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class PolicyFrame:
    """base_link -> the sim's world: p_w = Rz(yaw) p_b + (offset_x, offset_y, -table_z).

    yaw and offset default to zero: the blocks then have to be where they were in training, x 0.38 to 0.60 m
    and |y| < 0.18 m in front of the base. finger_axis_yaw is the direction the real fingers close along at
    tool yaw 0 (tower.yaml), so the sim yaw is the tool yaw plus it.
    """
    table_z: float
    yaw: float = 0.0
    offset: tuple = (0.0, 0.0)
    finger_axis_yaw: float = 0.0

    def _shift(self) -> np.ndarray:
        return np.array([self.offset[0], self.offset[1], -self.table_z])

    def to_world(self, p_base) -> np.ndarray:
        return rot_z(self.yaw) @ np.asarray(p_base, float) + self._shift()

    def to_base(self, p_world) -> np.ndarray:
        return rot_z(-self.yaw) @ (np.asarray(p_world, float) - self._shift())

    def vec_to_world(self, v_base) -> np.ndarray:
        return rot_z(self.yaw) @ np.asarray(v_base, float)

    def tcp_from_tool(self, tool: ToolPose):
        """Sim TCP position and yaw for a fingertip pose in base_link."""
        tcp = self.to_world(tool.xyz() + [0.0, 0.0, TIP_BELOW_TCP])
        return tcp, wrap_pi(tool.yaw + self.finger_axis_yaw + self.yaw)

    def tool_from_tcp(self, tcp, yaw: float) -> ToolPose:
        p = self.to_base(tcp) - [0.0, 0.0, TIP_BELOW_TCP]
        return ToolPose(float(p[0]), float(p[1]), float(p[2]), wrap_pi(yaw - self.finger_axis_yaw - self.yaw))

    def block(self, e: BlockEstimate, below: float, height: float):
        """World position (height/2 above the bottom) and yaw of a block standing `below` above the table
        measured around it; heights come from the ruler, not from the camera top (dark tops read low)."""
        return self.to_world([e.x, e.y, e.bottom + below + height / 2]), e.yaw + self.yaw


def stack_below(color: str, heights: dict) -> float:
    """Height of the blocks under `color` in the finished tower."""
    return sum(heights[b] for b in BLOCKS[:BLOCKS.index(color)])


def sensor_force(f_base, yaw_world: float, frame: PolicyFrame, sign: float = -1.0) -> np.ndarray:
    """A wrench force in base_link to the sim's wrist_force sensor axes (the TCP frame: z down, x along the
    fingers). MuJoCo reports the force the wrist exerts on the gripper, so pressing a held block down on the
    tower reads +z; sign -1 assumes the UR wrench is the force of the environment on the tool. Check on the
    arm: push the fingers up by hand, the logged policy force z has to go positive."""
    return sign * down_rotation(yaw_world).T @ frame.vec_to_world(f_base)


class StageState:
    """What TowerEnv knows during one stage, with its attribute names, so ScriptedStack can read it.

    Estimates are in the sim frame: est_held the held block's center, est_top the target's top face. The
    close-ups are taken before the stage starts and swapped in when the TCP meets the sim's wrist trigger
    (TowerEnv._wrist_look), so the policy sees them at the same moment as in training.
    """

    def __init__(self, cfg: SimConfig, stage: int, held: tuple, target: tuple, held_size, target_width: float,
                 tcp, yaw: float, held_close: tuple = None, target_close: tuple = None):
        self.cfg = cfg
        self.stage = stage
        self.est_held, self.est_held_yaw = np.asarray(held[0], float), float(held[1])
        self.est_top, self.est_target_yaw = np.asarray(target[0], float), float(target[1])
        self.est_size = np.asarray(held_size, float)
        self.est_target_width = float(target_width)
        self.held_close, self.target_close = held_close, target_close
        self.wrist_held = self.wrist_target = False
        self.carried = False
        self.lifted = False
        self.target_pos = np.asarray(tcp, float).copy()
        self.target_yaw = float(yaw)
        self.force = np.zeros(3)
        self.prev_action = np.zeros(5)
        self.t = 0
        self.update_targets()

    @property
    def unwrapped(self):
        return self

    def update_targets(self):
        above = self.cfg.finger_above_bottom
        if self.cfg.palm_above_top is not None:
            above = max(above, self.est_size[1] - FINGER_LEN + self.cfg.palm_above_top)
        self.grasp_z_off = TIP_BELOW_TCP + above
        self.est_grasp = self.est_held + [0, 0, self.grasp_z_off - self.est_size[1] / 2]
        self.est_place = self.est_top + [0, 0, self.grasp_z_off]

    def wrist_look(self, tcp):
        """TowerEnv._wrist_look: within wrist_radius (xy) of an estimate and at most wrist_max_height above it,
        the close-up replaces the first estimate; the held block only before it is lifted."""
        cfg = self.cfg
        if not cfg.wrist_camera:
            return
        tcp = np.asarray(tcp, float)

        def above(top):
            h = tcp[2] - TIP_BELOW_TCP - top[2]
            return np.linalg.norm(tcp[:2] - top[:2]) < cfg.wrist_radius and -0.005 < h < cfg.wrist_max_height

        if not self.wrist_held and not self.lifted and above(self.est_held + [0, 0, self.est_size[1] / 2]):
            if self.held_close is not None:
                self.est_held, self.est_held_yaw = np.asarray(self.held_close[0], float), float(self.held_close[1])
            self.wrist_held = True
        if not self.wrist_target and above(self.est_top):
            if self.target_close is not None:
                self.est_top, self.est_target_yaw = (np.asarray(self.target_close[0], float),
                                                     float(self.target_close[1]))
            self.wrist_target = True
        self.update_targets()

    def obs(self, tcp, yaw: float, opening: float) -> np.ndarray:
        """TowerEnv._obs, 29 values. prev_action[4] must be the gripper command as applied (+-1 after the
        binary gripper), as TowerEnv saw it in training: closed below 0."""
        tcp = np.asarray(tcp, float)
        self.carried = carry_latch(self.carried, self.prev_action[4] < 0, tcp, self.est_grasp)
        if self.cfg.carry_in_obs:
            slot27 = float(self.carried)
        else:
            slot27 = self.t / self.cfg.episode_steps if self.cfg.time_in_obs else 0.0
        return np.concatenate([
            tcp - OBS_CENTER,
            [math.cos(yaw), math.sin(yaw)],
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
            [slot27],
            [self.stage],
        ]).astype(np.float32)


def residual_obs(obs: np.ndarray, base_action, phase: int) -> np.ndarray:
    """ResidualTowerEnv._augment: the scripted action and its phase one-hot after the 29 TowerEnv values."""
    one_hot = np.zeros(N_PHASES)
    one_hot[min(phase, N_PHASES - 1)] = 1.0
    return np.concatenate([obs, base_action, one_hot]).astype(np.float32)


def total_action(base_action, residual, scale=RESIDUAL_SCALE) -> np.ndarray:
    return np.clip(np.asarray(base_action, float) + np.asarray(scale) * np.asarray(residual, float), -1.0, 1.0)


def step_target(pos, yaw: float, action, cfg: SimConfig, max_step: float = math.inf,
                max_yaw_step: float = math.inf):
    """TowerEnv.step's target update, with a speed cap: the displacement is shortened to max_step and the
    yaw increment to max_yaw_step before the sim's workspace clip."""
    a = np.clip(np.asarray(action, float), -1.0, 1.0)
    d = a[:3] * cfg.max_dpos
    n = float(np.linalg.norm(d))
    if n > max_step:
        d *= max_step / n
    dyaw = float(np.clip(a[3] * cfg.max_dyaw, -max_yaw_step, max_yaw_step))
    return np.clip(np.asarray(pos, float) + d, cfg.target_lo, cfg.target_hi), wrap_pi(yaw + dyaw)


def sim_start(held_center, held_height: float, tallest_top: float, cfg: SimConfig) -> np.ndarray:
    """Start TCP as TowerEnv._randomize draws it, without the random part: over the held block, the middle of
    start_above over the highest block."""
    z = tallest_top + TIP_BELOW_TCP + float(np.mean(cfg.start_above))
    return np.clip(np.array([held_center[0], held_center[1], z]), cfg.target_lo, cfg.target_hi)


def stage_done(script, state: StageState, rise: float = 0.03) -> bool:
    """ScriptedStack is in its last phase (released, going back up) and the TCP target is `rise` above the
    release point: the sim counts a placement once the gripper is open and clear for three steps."""
    return script.phase == 7 and script.hold_pos is not None and state.target_pos[2] >= script.hold_pos[2] + rise


class GripperModel:
    """/toggle_gripper only toggles and reports no width. The sim's gripper action is a finger speed whose sign
    the script sets (+-1, the residual moves it by at most 0.3), so: close below -threshold, open above
    +threshold. The opening the policy sees is the sim's finger position for the state: 1 open, the block
    width when closed on it, 0 closed on nothing."""

    def __init__(self, threshold: float = 0.5, closed: bool = False):
        self.threshold = threshold
        self.closed = closed
        self.width = 0.0

    def wanted(self, g: float):
        """True to close, False to open, None to leave it."""
        if not self.closed and g < -self.threshold:
            return True
        if self.closed and g > self.threshold:
            return False
        return None

    def set(self, closed: bool, width: float = 0.0):
        self.closed = closed
        self.width = width if closed else 0.0

    @property
    def opening(self) -> float:
        if not self.closed:
            return 1.0
        return min(1.0, self.width / 2 / MAX_OPEN)


def closing_on_block(tcp, est_grasp, tol: float = 0.015) -> bool:
    """The fingers close around the held block when the TCP is at the grasp estimate."""
    return bool(np.all(np.abs(np.asarray(tcp, float) - est_grasp) < tol))


def carry_latch(carried: bool, closed: bool, tcp, est_grasp, tol: float = 0.015) -> bool:
    """pensig.tower.carry_latch: set when closed at the grasp estimate, cleared when opened there."""
    if closing_on_block(tcp, est_grasp, tol):
        return bool(closed)
    return carried


# ---------------------------------------------------------------- the trained network

class NumpyPolicy:
    """The PPO actor as exported by scripts/export_tower_policy.py: VecNormalize statistics, tanh MLP and the
    deterministic action (the Gaussian mean, clipped to [-1, 1] as SB3's predict does). Runs without torch."""

    def __init__(self, path):
        d = np.load(path)
        self.mean, self.var = d["obs_mean"], d["obs_var"]
        self.eps, self.clip = float(d["epsilon"]), float(d["clip_obs"])
        self.norm = bool(d["norm_obs"])
        n = int(d["n_layers"])
        self.layers = [(d[f"w{k}"], d[f"b{k}"]) for k in range(n)]

    def normalize(self, obs) -> np.ndarray:
        obs = np.asarray(obs, np.float64)
        if not self.norm:
            return obs
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)

    def __call__(self, obs) -> np.ndarray:
        x = self.normalize(obs)
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)


def load_policy(path):
    """policy.npz (numpy, no torch), or final_model.zip with its vecnormalize.pkl as scripts/evaluate.py."""
    path = Path(path)
    if path.suffix == ".npz":
        return NumpyPolicy(path)
    import pickle

    from stable_baselines3 import PPO

    model = PPO.load(path, device="cpu")
    with open(path.parent / "vecnormalize.pkl", "rb") as f:
        norm = pickle.load(f)

    def act(obs):
        obs = np.clip((obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + norm.epsilon),
                      -norm.clip_obs, norm.clip_obs)
        return model.predict(obs, deterministic=True)[0]

    return act


REPO_CANDIDATES = ("~/ur7e-block-tower", "~/ur-pen-signature")


def resolve_repo(path: str = "") -> Path:
    """The `repo` parameter if set, else the first existing checkout: ~/ur7e-block-tower, then
    ~/ur-pen-signature (the repository's earlier name, still the checkout on the lab machine)."""
    if path:
        return Path(path).expanduser()
    for candidate in REPO_CANDIDATES:
        p = Path(candidate).expanduser()
        if p.exists():
            return p
    return Path(REPO_CANDIDATES[0]).expanduser()


def scripted_stack_class(repo=None):
    """pensig.tower_scripted.ScriptedStack, the script itself. pensig.tower needs MuJoCo, and ScriptedStack
    only takes TIP_BELOW_TCP and wrap_quarter from it, so without MuJoCo those two come from here."""
    if repo is not None and str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import pensig  # noqa: F401  (pensig not found: set `repo` to the ur7e-block-tower checkout)

    try:
        import pensig.tower  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pensig.tower")
        stub.TIP_BELOW_TCP = TIP_BELOW_TCP
        stub.wrap_quarter = wrap_quarter
        sys.modules["pensig.tower"] = stub
    from pensig.tower_scripted import ScriptedStack

    return ScriptedStack

