"""Tower episodes that start partway through the task, for an end-to-end policy (no script under it at run time).

PPO from the start of the task never finds a placement, so it never sees a success to learn from. Here the
scripted stack plays the first part of the episode and hands over at a given phase: at level 0 the policy only
has to let go and lift (phase 7), each level adds one earlier phase, and from level 8 some episodes start from
the plain reset, as at run time. Start states from demonstrations: Nair et al. 2018, Salimans and Chen 2018;
the curriculum on start states: Florensa et al. 2017.

Block sizes are drawn at every reset, so one policy has to handle any block within the gripper stroke.

Replaying the script at every reset stalled the 32 lockstep environments of SubprocVecEnv behind whichever was
replaying (43 steps/s on Savio instead of ~1300). One replay now runs the script to the end, saves about
twenty states along the way, and later resets restore them, each a few times.
"""

import copy
from collections import defaultdict

import gymnasium as gym
import mujoco
import numpy as np

from pensig.tower_scripted import ScriptedStack

N_PHASES = 8  # ScriptedStack phases 0..7
MAX_LEVEL = N_PHASES + 1  # 8: full starts join, 9: half of the starts are full starts


def random_sizes(rng) -> dict:
    """Square blocks, (width, height): the base wider, the held ones within the 50 mm Hand-E stroke."""
    return {"base_size": (rng.uniform(0.040, 0.060), rng.uniform(0.025, 0.080)),
            "mid_size": (rng.uniform(0.025, 0.044), rng.uniform(0.025, 0.080)),
            "top_size": (rng.uniform(0.025, 0.044), rng.uniform(0.025, 0.060))}


# What a reset changes in the model (TowerEnv._randomize) and in the data, plus the env's own state.
MODEL_FIELDS = ("geom_size", "geom_aabb", "geom_rbound", "geom_friction", "body_mass", "body_inertia", "body_ipos",
                "body_iquat", "actuator_forcerange")
DATA_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")
STATIC = {"model", "data", "ik", "render_mode", "n_sub", "finger_qadr", "qadr", "dadr", "arm_act", "grip_act",
          "tcp_sid", "ft_adr", "gid", "bid", "gripper_geoms", "pad_geoms", "action_space", "observation_space",
          "_renderer", "_camera", "_np_random", "_np_random_seed"}


def snapshot(e) -> dict:
    return {"model": {f: getattr(e.model, f).copy() for f in MODEL_FIELDS},
            "data": {f: getattr(e.data, f).copy() for f in DATA_FIELDS}, "time": e.data.time,
            "attrs": {k: copy.deepcopy(v) for k, v in vars(e).items() if k not in STATIC}}


def restore(e, s: dict):
    for f, v in s["model"].items():
        getattr(e.model, f)[:] = v
    for f, v in s["data"].items():
        getattr(e.data, f)[:] = v
    e.data.time = s["time"]
    for k, v in s["attrs"].items():
        setattr(e, k, copy.deepcopy(v))
    mujoco.mj_forward(e.model, e.data)


def binary_grip(env, action, threshold: float = 0.5) -> np.ndarray:
    """The gripper as on the lab arm (/toggle_gripper, tower_ros.policy.GripperModel): closed at full speed once the
    command goes below -threshold, open above +threshold, unchanged in between. With the sim's finger speed taken
    as is, PPO exploring around -0.3 got weak grips that slipped, and learned that closing on a block fails.
    Stateless: the last applied command (TowerEnv.prev_action) holds the state, so restored states keep theirs."""
    a = np.clip(np.array(action, dtype=np.float64), -1.0, 1.0)
    closed = env.unwrapped.prev_action[4] < 0
    if a[4] < -threshold:
        closed = True
    elif a[4] > threshold:
        closed = False
    a[4] = -1.0 if closed else 1.0
    return a


def frontier(level: int) -> int:
    """The earliest start phase at this level, -1 for the plain reset."""
    return N_PHASES - 1 - level if level < N_PHASES else -1


class ScriptStart(gym.Wrapper):
    """TowerEnv whose episodes start where the script hands over. info["start_phase"]: -1 for the plain reset."""

    def __init__(self, env, sizes: dict | None = None, noise=(0.0, 0.1), save_prob: float = 0.25, uses: int = 6,
                 pool: int = 40, seed: int = 0):
        super().__init__(env)
        self.sizes = sizes  # None: random sizes at every reset
        self.noise = noise
        self.save_prob = save_prob  # chance to save the state at a step, besides the first step of each phase
        self.uses = uses  # resets per saved state
        self.rng = np.random.default_rng(seed)
        self.script = ScriptedStack()
        self.pool = defaultdict(list)  # phase -> [[uses left, env state, script state]]
        self.pool_size = pool
        self.level = 0
        self.fixed_phase = None  # set: every episode starts at this phase (evaluation)
        self.weights = None  # per start phase -1..7: more episodes where the policy fails more
        self.start_phase = -1

    def set_level(self, level: int):
        self.level = int(level)

    def set_weights(self, weights):
        self.weights = None if weights is None else np.asarray(weights, float)

    def _sample_phase(self) -> int:
        if self.fixed_phase is not None:
            return self.fixed_phase
        if self.level >= MAX_LEVEL and self.rng.random() < 0.5:
            return -1
        lo = frontier(min(self.level, N_PHASES))  # lo = -1 includes the plain reset
        if self.weights is None:
            return int(self.rng.integers(lo, N_PHASES))
        w = self.weights[lo + 1:]
        return int(lo + self.rng.choice(len(w), p=w / w.sum()))

    def _new_sizes(self):
        for k, v in (self.sizes or random_sizes(self.rng)).items():
            setattr(self.env.unwrapped.cfg, k, v)

    def reset(self, *, seed=None, options=None):
        e = self.env.unwrapped
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p = self._sample_phase()
        for _ in range(10):
            if p < 0 or self.pool[p]:
                break
            self._replay(seed, options)
            seed = None
        if p >= 0 and self.pool[p]:
            k = int(self.rng.integers(len(self.pool[p])))
            entry = self.pool[p][k]
            restore(e, entry[1])
            vars(self.script).update(copy.deepcopy(entry[2]))  # only for checks that let the script go on
            entry[0] -= 1
            if entry[0] <= 0:
                self.pool[p].pop(k)
            self.start_phase = p
            e.t = 0  # the policy gets the full step budget from the handover
            obs, info = e._obs(), e._info(False)
        else:
            self._new_sizes()
            obs, info = self.env.reset(seed=seed, options=options)
            self.start_phase = -1
        info["start_phase"] = self.start_phase
        return obs, info

    def _replay(self, seed, options):
        """One episode driven by the script, saving states along the way into the pool."""
        e = self.env.unwrapped
        self._new_sizes()
        self.env.reset(seed=seed, options=options)
        self.script.reset()
        noise = self.rng.choice(self.noise)
        last = -1
        for _ in range(e.cfg.episode_steps):
            a = self.script(e)  # updates the phase from the current state
            ph = self.script.phase
            if ph != last or self.rng.random() < self.save_prob:
                if len(self.pool[ph]) >= self.pool_size:
                    self.pool[ph].pop(int(self.rng.integers(len(self.pool[ph]))))
                self.pool[ph].append([self.uses, snapshot(e), copy.deepcopy(vars(self.script))])
                last = ph
            a[:4] = np.clip(a[:4] + noise * self.rng.standard_normal(4), -1, 1)
            _, _, term, trunc, _ = self.env.step(a)
            if term or trunc:
                return

    def script_episodes(self, n: int) -> tuple[dict, int]:
        """n episodes driven by the script from the plain reset, random sizes, as replay transitions (obs, act,
        rew, next, done: terminated or out of time). Returns them and the number of successes."""
        e = self.env.unwrapped
        out = {k: [] for k in ("obs", "act", "rew", "next", "done")}
        wins = 0
        for _ in range(n):
            self._new_sizes()
            obs, _ = self.env.reset()
            self.script.reset()
            noise = self.rng.choice(self.noise)
            done = False
            while not done:
                a = self.script(e)
                a[:4] = np.clip(a[:4] + noise * self.rng.standard_normal(4), -1, 1)
                a = binary_grip(self.env, a)
                nxt, r, term, trunc, info = self.env.step(a)
                done = term or trunc
                for k, v in zip(out, (obs, a, r, nxt, done)):
                    out[k].append(v)
                obs = nxt
            wins += bool(info["is_success"])
        return {k: np.asarray(v, np.float32) for k, v in out.items()}, wins

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(binary_grip(self.env, action))
        info["start_phase"] = self.start_phase
        return obs, r, term, trunc, info
