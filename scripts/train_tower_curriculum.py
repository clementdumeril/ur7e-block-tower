"""End-to-end tower policy: PPO on TowerEnv alone, episodes started partway by the script (pensig.curriculum).

The level goes up once the policy places in half of the last 200 episodes from every start phase in use, not
only the earliest: the first run (tower_cur_40966043) passed levels 3 and 4, then at level 5 forgot them (6 % from
phase 4, 0 % from phase 3 after 20M more steps). Start phases are also drawn more often where the policy fails.
That alone did not stop it (tower_cur_40978032: phases 2-4 fell to 1-16 % at level 5 while 5-7 stayed above
80 %): with gamma 0.98 and no cost for running out of time, hovering until the time limit was worth as much as a
grasp 70 steps from success. Hence gamma 0.995 and a timeout penalty equal to the failure penalty.
That did not stop it either (tower_cur_41005533: phases 2-3 at 0 %), so the gripper is now open/closed as on the
lab arm (pensig.curriculum.binary_grip), and the log has the failure reasons per start phase and the action std.
Block sizes are random at every reset. At the end, full towers from the plain reset, the two stages chained and
no script anywhere, on the measured lab blocks and on random ones.

python scripts/train_tower_curriculum.py --out runs/tower_cur --timesteps 20000000
python scripts/train_tower_curriculum.py --resume runs/tower_cur --out runs/tower_cur2   # newest checkpoint, its level
python scripts/train_tower_curriculum.py --eval-only runs/tower_cur/final_model.zip
python scripts/train_tower_curriculum.py --eval-phases runs/tower_cur   # success and failures per start phase
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import load_policy  # noqa: E402
from pensig.config import TowerConfig  # noqa: E402
from pensig.curriculum import MAX_LEVEL, N_PHASES, ScriptStart, binary_grip, frontier, random_sizes  # noqa: E402
from pensig.tower import TowerEnv  # noqa: E402

MEASURED = {"base_size": (0.05, 0.05), "mid_size": (0.04, 0.08), "top_size": (0.035, 0.03)}


def make_env(rank: int, seed: int, level: int, gamma: float, timeout_penalty: float):
    def thunk():
        cfg = TowerConfig()
        cfg.shaping_gamma, cfg.timeout_penalty = gamma, timeout_penalty
        env = ScriptStart(TowerEnv(cfg), seed=seed + 1000 * rank)
        env.set_level(level)
        env.reset(seed=seed + 1000 * rank)
        return env
    return thunk


def resume_from(run: Path):
    """Newest checkpoint of a run (or its final model), its VecNormalize stats, and the level it had reached.
    A checkpoint zip instead of a run: that checkpoint, level from --level."""
    if run.suffix == ".zip":
        n = run.stem.split("_")[1]
        return run, run.parent / f"ppo_vecnormalize_{n}_steps.pkl", None
    final = run / "final_model.zip"
    ckpts = sorted((run / "checkpoints").glob("ppo_*_steps.zip"), key=lambda f: int(f.stem.split("_")[1]))
    if final.exists():
        zip_, stats = final, run / "vecnormalize.pkl"
    elif ckpts:
        n = ckpts[-1].stem.split("_")[1]
        zip_, stats = ckpts[-1], run / "checkpoints" / f"ppo_vecnormalize_{n}_steps.pkl"
    else:
        raise FileNotFoundError(f"no model in {run}")
    log = json.loads((run / "curriculum.json").read_text()) if (run / "curriculum.json").exists() else []
    return zip_, stats, (log[-1]["level"] if log else 0)


def evaluate_phases(run: Path, episodes: int, seed: int = 20_000) -> dict:
    """The newest model of a run, started at each script phase in turn (random sizes): where does it fail?"""
    zip_, stats, level = resume_from(run) if run.is_dir() else (run, run.parent / "vecnormalize.pkl", None)
    policy = load_policy(zip_)
    w = ScriptStart(TowerEnv(TowerConfig()), seed=seed)
    out = {"model": str(zip_), "level": level}
    for p in [*range(N_PHASES - 1, -1, -1), -1]:
        w.fixed_phase = p
        ok, why, steps = 0, {}, 0
        for ep in range(episodes):
            obs, _ = w.reset(seed=seed + ep if ep == 0 else None)
            done = False
            while not done:
                obs, _, term, trunc, info = w.step(policy(w, obs))
                done, steps = term or trunc, steps + 1
            ok += bool(info["is_success"])
            if not info["is_success"]:
                r = info["failure"] or ("stop" if info.get("stopped") else "unstable" if info.get("unstable")
                                        else "timeout")
                why[r] = why.get(r, 0) + 1
        out[f"phase {p}" if p >= 0 else "plain reset"] = {"success": ok / episodes, "failures": why,
                                                           "mean_steps": round(steps / episodes)}
        print(f"start {p}: {ok / episodes:.2f} success, failures {why}, {steps / episodes:.0f} steps", flush=True)
    return out


def evaluate(path: Path, towers: int, seed: int = 10_000) -> dict:
    """Full towers from the plain reset, the policy alone, on the measured blocks and on random ones."""
    policy = load_policy(path)
    env = TowerEnv(TowerConfig())
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("measured", "random"):
        built, stage_ok, failures = [], [[], []], {}
        for ep in range(towers):
            for k, v in (MEASURED if mode == "measured" else random_sizes(rng)).items():
                setattr(env.cfg, k, v)
            obs, _ = env.reset(seed=seed + ep, options={"stage": 0})
            n = 0
            for stage in (0, 1):
                if stage == 1:
                    obs, _ = env.next_stage()
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(binary_grip(env, policy(env, obs)))
                    done = term or trunc
                stage_ok[stage].append(bool(info["is_success"]))
                if not info["is_success"]:
                    why = f"stage{stage}:" + (info["failure"] or ("stop" if info.get("stopped") else "timeout"))
                    failures[why] = failures.get(why, 0) + 1
                    break
                n += 1
            built.append(n == 2)
        out[mode] = {"full_tower": float(np.mean(built)), "stage0": float(np.mean(stage_ok[0])),
                     "stage1": float(np.mean(stage_ok[1])) if stage_ok[1] else None, "failures": failures}
        print(f"eval {mode} blocks: {out[mode]}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/tower_cur"))
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=os.cpu_count())
    ap.add_argument("--level", type=int, help="curriculum level to start at (default 0, or the resumed run's)")
    ap.add_argument("--resume", type=Path, help="run directory to continue from: newest checkpoint and its level")
    ap.add_argument("--threshold", type=float, default=0.5, help="success from the earliest start to level up")
    ap.add_argument("--window", type=int, default=200, help="episodes from the earliest start that count")
    ap.add_argument("--log-std-init", type=float, default=-1.2)
    ap.add_argument("--gamma", type=float, default=0.995, help="PPO discount, also the reward shaping one")
    ap.add_argument("--timeout-penalty", type=float, default=3.0, help="reward lost when the steps run out")
    ap.add_argument("--max-minutes", type=float, help="stop training after this long, then save and evaluate")
    ap.add_argument("--eval-towers", type=int, default=100)
    ap.add_argument("--eval-only", type=Path, help="model zip: only evaluate it")
    ap.add_argument("--eval-phases", type=Path, help="run directory or model zip: success from each start phase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.eval_phases:
        res = evaluate_phases(args.eval_phases, args.eval_towers)
        if args.eval_phases.is_dir():
            (args.eval_phases / "eval_phases.json").write_text(json.dumps(res, indent=2))
        return
    if args.eval_only:
        evaluate(args.eval_only, args.eval_towers)
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({"env": asdict(TowerConfig()), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2, default=str))

    resume = resume_from(args.resume) if args.resume else None
    if args.level is None:
        args.level = (resume[2] if resume else 0) or 0
    venv = SubprocVecEnv([make_env(i, args.seed, args.level, args.gamma, args.timeout_penalty)
                          for i in range(args.n_envs)])
    if resume:
        print(f"resuming from {resume[0]} at level {args.level}", flush=True)
        venv = VecNormalize.load(str(resume[1]), venv)
        venv.gamma = args.gamma
        model = PPO.load(resume[0], env=venv, device="cpu", gamma=args.gamma)
        model.tensorboard_log = str(args.out)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
        model = PPO("MlpPolicy", venv, n_steps=512, batch_size=1024, n_epochs=8, learning_rate=3e-4, gamma=args.gamma,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=args.seed, device="cpu", verbose=0,
                    tensorboard_log=str(args.out),
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=args.log_std_init))

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.level = args.level
            self.phase = {p: deque(maxlen=args.window) for p in range(-1, N_PHASES)}  # successes per start phase
            self.why = {p: deque(maxlen=args.window) for p in range(-1, N_PHASES)}  # outcome per start phase
            self.all, self.log, self.t0 = deque(maxlen=1000), [], time.time()

        def active(self):
            return range(frontier(min(self.level, MAX_LEVEL - 1)), N_PHASES)

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    ok, p = bool(info.get("is_success")), info.get("start_phase", -1)
                    self.all.append(ok)
                    self.phase[p].append(ok)
                    self.why[p].append("ok" if ok else info.get("failure") or (
                        "stop" if info.get("stopped") else "unstable" if info.get("unstable") else "timeout"))
            act = self.active()
            if (self.level < MAX_LEVEL and all(len(self.phase[p]) == args.window for p in act)
                    and min(np.mean(self.phase[p]) for p in act) >= args.threshold):
                self.level += 1
                self.phase[frontier(min(self.level, MAX_LEVEL - 1))].clear()
                venv.env_method("set_level", self.level)
                self._print(f"level up -> {self.level}")
            return not args.max_minutes or time.time() - self.t0 < 60 * args.max_minutes

        def weights(self):
            """Start phases -1..7: weight 0.2 + failure rate, unknown phases as if always failed."""
            return [0.2 + 1.0 - (float(np.mean(self.phase[p])) if self.phase[p] else 0.0)
                    for p in range(-1, N_PHASES)]

        def _print(self, what=""):
            mean = lambda q: round(float(np.mean(q)), 2) if q else None  # noqa: E731
            entry = {"steps": self.num_timesteps, "level": self.level, "all_success": mean(self.all),
                     "success_by_start": {p: mean(self.phase[p]) for p in self.active()},
                     "outcomes_by_start": {p: {k: self.why[p].count(k) for k in sorted(set(self.why[p]))}
                                           for p in self.active()},
                     "action_std": [round(float(x), 2) for x in model.policy.log_std.exp().detach().numpy()],
                     "minutes": round((time.time() - self.t0) / 60, 1)}
            self.log.append(entry)
            print(json.dumps(entry), what, flush=True)
            (args.out / "curriculum.json").write_text(json.dumps(self.log, indent=1))

        def _on_rollout_end(self) -> None:
            venv.env_method("set_weights", self.weights())
            if self.n_calls % (20 * 512) < 512:  # about every 20 rollouts
                self._print()
            venv.save(str(args.out / "vecnormalize.pkl"))

    cur = Curriculum()
    checkpoints = CheckpointCallback(save_freq=max(1_000_000 // args.n_envs, 1),
                                     save_path=str(args.out / "checkpoints"), name_prefix="ppo",
                                     save_vecnormalize=True)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoints, cur], reset_num_timesteps=not resume)
    cur._print("end of training")
    model.save(args.out / "final_model")
    venv.save(str(args.out / "vecnormalize.pkl"))
    venv.close()

    result = {"curriculum": cur.log[-1], "eval": evaluate(args.out / "final_model.zip", args.eval_towers)}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
