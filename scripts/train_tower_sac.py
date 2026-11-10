"""End-to-end tower policy with SAC and the scripted stack's episodes in the replay buffer (pensig.rlpd).

Same environment as scripts/train_tower_curriculum.py: TowerEnv alone at run time, episodes started partway by
the script with the same level gate, random block sizes, open/closed gripper. What changes is the learner: PPO
lost the grasp phases at level 5 in four runs, with its action std down to 0.07. Before training, the workers
play `--demos` script episodes from the plain reset; half of every SAC batch is drawn from them.
Timeouts end the episode without bootstrapping: the step count is in the observation.
Gradient steps are the cost on CPU (about 40/s here for a 256x256 SAC, batch 256), hence UTD 0.25 by default.
With demonstrations in the buffer only, the grasp phases were lost again (tower_sac_41035915); the actor now
starts as a behaviour-cloned copy of the script (`--bc-pretrain`) and keeps a cloning term on the demonstration
half of every batch (`--bc`, pensig.rlpd.DemoSAC).

python scripts/train_tower_sac.py --out runs/tower_sac --timesteps 20000000
python scripts/train_tower_sac.py --resume runs/tower_sac/checkpoints/sac_5000000_steps.zip --level 4 --out runs/s2
python scripts/train_tower_sac.py --eval-only runs/tower_sac/final_model.zip
python scripts/train_tower_sac.py --eval-phases runs/tower_sac/final_model.zip
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_tower_curriculum import evaluate, evaluate_phases, make_env  # noqa: E402
from pensig.config import TowerConfig  # noqa: E402
from pensig.curriculum import MAX_LEVEL, N_PHASES, ScriptStart, frontier  # noqa: E402
from pensig.tower import TowerEnv  # noqa: E402


def stats_for(path: Path) -> Path:
    """VecNormalize stats saved with a model: vecnormalize.pkl, or sac_vecnormalize_N_steps.pkl for a checkpoint."""
    parts = path.stem.split("_")
    if parts[0] == "sac" and len(parts) == 3:
        return path.parent / f"sac_vecnormalize_{parts[1]}_steps.pkl"
    return path.parent / "vecnormalize.pkl"


def load_sac(path: Path):
    from pensig.rlpd import DemoSAC

    model = DemoSAC.load(path, device="cpu")
    with open(stats_for(path), "rb") as f:
        norm = pickle.load(f)

    def act(env, obs):
        obs = np.clip((obs - norm.obs_rms.mean) / np.sqrt(norm.obs_rms.var + norm.epsilon),
                      -norm.clip_obs, norm.clip_obs)
        return model.predict(obs, deterministic=True)[0]

    act.reset = lambda: None
    return act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/tower_sac"))
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=os.cpu_count())
    ap.add_argument("--level", type=int, default=0, help="curriculum level to start at")
    ap.add_argument("--resume", type=Path, help="SAC model zip to continue from (its VecNormalize stats next to it)")
    ap.add_argument("--demos", type=int, default=2000, help="script episodes for the demonstration buffer")
    ap.add_argument("--demo-ratio", type=float, default=0.5, help="share of every batch from the demonstrations")
    ap.add_argument("--bc", type=float, default=0.4, help="weight of the cloning term (TD3+BC: 0.4), 0: plain SAC")
    ap.add_argument("--bc-grip", type=float, default=0.0, help="gripper weight in the cloning term after pre-training")
    ap.add_argument("--det", type=float, default=0.0, help="weight of -Q at the mean action in the actor loss")
    ap.add_argument("--q-filter", action="store_true", help="clone only where the critic prefers the script")
    ap.add_argument("--bc-pretrain", type=int, default=50_000, help="actor-only cloning steps before training")
    ap.add_argument("--utd", type=float, default=0.25, help="gradient steps per environment step")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--net", type=int, nargs="+", default=[256, 256], help="hidden layers, actor and critics")
    ap.add_argument("--buffer", type=int, default=3_000_000)
    ap.add_argument("--threads", type=int, default=8, help="torch threads for the gradient steps")
    ap.add_argument("--threshold", type=float, default=0.5, help="success from every start phase in use to level up")
    ap.add_argument("--gate-every", type=int, default=200_000, help="steps between deterministic gate checks")
    ap.add_argument("--gate-episodes", type=int, default=20, help="deterministic episodes per start phase")
    ap.add_argument("--window", type=int, default=200, help="episodes per start phase that count")
    ap.add_argument("--gamma", type=float, default=0.99, help="SAC discount, also the reward shaping one")
    ap.add_argument("--timeout-penalty", type=float, default=3.0, help="reward lost when the steps run out")
    ap.add_argument("--max-minutes", type=float, help="stop training after this long, then save and evaluate")
    ap.add_argument("--eval-towers", type=int, default=100)
    ap.add_argument("--eval-only", type=Path, help="model zip: only evaluate it")
    ap.add_argument("--eval-phases", type=Path, help="model zip: success and failures from each start phase")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.eval_phases:
        res = evaluate_phases(args.eval_phases, args.eval_towers, loader=load_sac)
        (args.eval_phases.parent / "eval_phases.json").write_text(json.dumps(res, indent=2))
        return
    if args.eval_only:
        evaluate(args.eval_only, args.eval_towers, loader=load_sac)
        return

    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from pensig.rlpd import DemoMixBuffer, DemoSAC, LNSACPolicy

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({"env": asdict(TowerConfig()), "args": {
        k: str(v) for k, v in vars(args).items()}}, indent=2, default=str))

    venv = SubprocVecEnv([make_env(i, args.seed, args.level, args.gamma, args.timeout_penalty)
                          for i in range(args.n_envs)])

    t0 = time.time()
    per = -(-args.demos // args.n_envs)
    parts = venv.env_method("script_episodes", per)
    demo = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    wins = sum(p[1] for p in parts)
    print(f"demonstrations: {per * args.n_envs} script episodes, {wins / (per * args.n_envs):.2f} success, "
          f"{len(demo['obs'])} transitions, {(time.time() - t0) / 60:.1f} min", flush=True)

    if args.resume:
        venv = VecNormalize.load(str(stats_for(args.resume)), venv)
        venv.norm_reward = False
        model = DemoSAC.load(args.resume, env=venv, device="cpu")
        model.learning_starts = model.num_timesteps + 20_000  # refill the online buffer first
        model.tensorboard_log = str(args.out)
        model.bc_weight, model.q_filter, model.bc_stats = args.bc, args.q_filter, {"bc": [], "kept": []}
        model.bc_grip, model.det_weight = args.bc_grip, args.det
        print(f"resuming from {args.resume} at level {args.level}", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=args.gamma)
        venv.obs_rms.update(demo["obs"])
        model = DemoSAC(LNSACPolicy, venv, buffer_size=args.buffer, batch_size=args.batch, learning_rate=3e-4,
                        gamma=args.gamma, tau=0.005, learning_starts=20_000, train_freq=1,
                        gradient_steps=max(1, round(args.utd * args.n_envs)), ent_coef="auto",
                        replay_buffer_class=DemoMixBuffer,
                        replay_buffer_kwargs=dict(demo_ratio=args.demo_ratio, handle_timeout_termination=False),
                        policy_kwargs=dict(net_arch=args.net), seed=args.seed, device="cpu", verbose=0,
                        tensorboard_log=str(args.out), bc_weight=args.bc, q_filter=args.q_filter,
                        bc_grip=args.bc_grip, det_weight=args.det)
    model.replay_buffer.set_demos(demo["obs"], demo["act"], demo["rew"], demo["next"], demo["done"], demo["clean"])
    if args.bc_pretrain and not args.resume:
        t0 = time.time()
        loss = model.pretrain_bc(args.bc_pretrain, args.batch)
        print(f"behaviour cloning: {args.bc_pretrain} steps, loss {loss:.4f}, {(time.time() - t0) / 60:.1f} min",
              flush=True)
        model.save(args.out / "bc_model")
        venv.save(str(args.out / "bc_vecnormalize.pkl"))

    class Curriculum(BaseCallback):
        def __init__(self):
            super().__init__()
            self.level = args.level
            self.phase = {p: deque(maxlen=args.window) for p in range(-1, N_PHASES)}  # successes per start phase
            self.why = {p: deque(maxlen=args.window) for p in range(-1, N_PHASES)}  # outcome per start phase
            self.all, self.log, self.t0 = deque(maxlen=1000), [], time.time()
            self.next_print = 0
            self.next_gate = args.gate_every
            self.det = {}
            cfg = TowerConfig()
            cfg.shaping_gamma, cfg.timeout_penalty = args.gamma, args.timeout_penalty
            # Every gate episode from its own script replay: one saved state per phase, used once. With the
            # training settings 20 episodes came from one or two replays, the same blocks, and the gate jumped
            # by 50 points between checks (tower_sac_41067442).
            self.gate_env = ScriptStart(TowerEnv(cfg), seed=args.seed + 99_000, save_prob=0.0, uses=1)

        def active(self):
            return range(frontier(min(self.level, MAX_LEVEL - 1)), N_PHASES)

        def gate(self) -> dict:
            """Success of the deterministic policy, the one the arm runs, from each start phase in use. The
            training episodes use sampled actions: in tower_sac_41037952 they succeeded 5-44 % from the grasp
            phases while the deterministic policy built the first block 79 % of the time from the plain reset,
            and the gate on them held the curriculum at level 6 for 12 hours."""
            w, out = self.gate_env, {}
            phases = [*self.active()]
            for p in phases:
                w.fixed_phase = p
                ok = 0
                for _ in range(args.gate_episodes):
                    obs, _ = w.reset()
                    done = False
                    while not done:
                        a = model.predict(venv.normalize_obs(obs), deterministic=True)[0]
                        obs, _, term, trunc, info = w.step(a)
                        done = term or trunc
                    ok += bool(info["is_success"])
                out[p] = round(ok / args.gate_episodes, 2)
            w.fixed_phase = None
            return out

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done:
                    ok, p = bool(info.get("is_success")), info.get("start_phase", -1)
                    self.all.append(ok)
                    self.phase[p].append(ok)
                    self.why[p].append("ok" if ok else info.get("failure") or (
                        "stop" if info.get("stopped") else "unstable" if info.get("unstable") else "timeout"))
            if self.num_timesteps >= self.next_gate and self.level < MAX_LEVEL:
                self.next_gate = self.num_timesteps + args.gate_every
                self.det = self.gate()
                if min(self.det.values()) >= args.threshold:
                    self.level += 1
                    self.phase[frontier(min(self.level, MAX_LEVEL - 1))].clear()
                    venv.env_method("set_level", self.level)
                    self._print(f"level up -> {self.level}")
            if self.n_calls % 200 == 0:
                venv.env_method("set_weights", self.weights())
            if self.num_timesteps >= self.next_print:
                self.next_print += 200_000
                self._print()
            return not args.max_minutes or time.time() - self.t0 < 60 * args.max_minutes

        def weights(self):
            """Start phases -1..7: weight 0.2 + failure rate, unknown phases as if always failed."""
            return [0.2 + 1.0 - (float(np.mean(self.phase[p])) if self.phase[p] else 0.0)
                    for p in range(-1, N_PHASES)]

        def _print(self, what=""):
            mean = lambda q: round(float(np.mean(q)), 2) if q else None  # noqa: E731
            std = None
            if model.replay_buffer.size() > 0:
                with torch.no_grad():
                    obs = model.replay_buffer.sample(1024, venv).observations
                    std = model.actor.get_action_dist_params(obs)[1].exp().mean(0)
                std = [round(float(x), 2) for x in std]
            entry = {"steps": self.num_timesteps, "level": self.level, "all_success": mean(self.all),
                     "success_by_start": {p: mean(self.phase[p]) for p in self.active()},
                     "deterministic_by_start": self.det,
                     "outcomes_by_start": {p: {k: self.why[p].count(k) for k in sorted(set(self.why[p]))}
                                           for p in self.active()},
                     "action_std": std, "alpha": round(float(model.log_ent_coef.detach().exp()), 4),
                     **model.pop_bc_stats(),
                     "minutes": round((time.time() - self.t0) / 60, 1)}
            self.log.append(entry)
            print(json.dumps(entry), what, flush=True)
            (args.out / "curriculum.json").write_text(json.dumps(self.log, indent=1))

    cur = Curriculum()
    checkpoints = CheckpointCallback(save_freq=max(1_000_000 // args.n_envs, 1),
                                     save_path=str(args.out / "checkpoints"), name_prefix="sac",
                                     save_vecnormalize=True)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoints, cur], reset_num_timesteps=not args.resume)
    cur._print("end of training")
    model.save(args.out / "final_model")
    venv.save(str(args.out / "vecnormalize.pkl"))
    venv.close()

    result = {"curriculum": cur.log[-1],
              "eval": evaluate(args.out / "final_model.zip", args.eval_towers, loader=load_sac)}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
