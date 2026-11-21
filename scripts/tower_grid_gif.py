"""Several towers side by side in one GIF: the end-to-end SAC policy (or the script) on consecutive seeds, not picked.

python scripts/tower_grid_gif.py --policy runs/tower_sac_41577803/best/best_model.zip --out media/tower_sac_measured.gif
python scripts/tower_grid_gif.py --policy runs/tower_sac_41577803/best/best_model.zip --sizes random --out media/tower_sac_random.gif
python scripts/tower_grid_gif.py --scripted --out media/tower_script.gif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_tower_curriculum import MEASURED  # noqa: E402
from train_tower_sac import load_sac  # noqa: E402
from pensig.config import TowerConfig  # noqa: E402
from pensig.curriculum import binary_grip, random_sizes  # noqa: E402
from pensig.tower import BLOCKS, TowerEnv  # noqa: E402
from pensig.tower_scripted import ScriptedStack  # noqa: E402


def view(env):
    """Wider than TowerEnv.render: the whole arm, not only the blocks."""
    if env._renderer is None:
        env.render()
    cam, base = env._camera, env.data.xpos[env.bid["green"]]
    cam.lookat[:] = (0.5 * (base[0] + 0.2), 0.5 * base[1], 0.30)
    cam.distance, cam.azimuth, cam.elevation = 1.45, 140, -24
    env._renderer.update_scene(env.data, cam)
    return env._renderer.render()


def play(env, act, sizes, seed, every):
    """One full tower from the plain reset; frames every `every` steps and the outcome."""
    for k, v in sizes.items():
        setattr(env.cfg, k, v)
    obs, _ = env.reset(seed=seed, options={"stage": 0})
    act.reset()
    frames, t, outcome = [view(env)], 0, "tower"
    for stage in (0, 1):
        if stage == 1:
            obs, _ = env.next_stage()
            act.reset()
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(env, obs))
            done, t = term or trunc, t + 1
            if t % every == 0:
                frames.append(view(env))
        if not info["is_success"]:
            outcome = info["failure"] or ("stop" if info.get("stopped") else "timeout")
            break
    frames.append(view(env))
    return frames, outcome


def label(frame, text, ok, scale):
    img = Image.fromarray(frame).resize((round(frame.shape[1] / scale), round(frame.shape[0] / scale)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    d.rectangle((0, 0, img.width, 22), fill=(20, 20, 20))
    d.text((6, 3), text, fill=(120, 230, 120) if ok else (240, 120, 120), font=font)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy", type=Path, help="SAC model zip (its VecNormalize stats next to it)")
    src.add_argument("--scripted", action="store_true")
    ap.add_argument("--sizes", choices=("measured", "random"), default="measured")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10_000, help="first seed; the next ones follow")
    ap.add_argument("--every", type=int, default=2, help="keep one frame every this many control steps")
    ap.add_argument("--scale", type=float, default=1.6, help="shrink each 640x480 view by this factor")
    ap.add_argument("--hold", type=int, default=15, help="frames to hold the end before the GIF loops")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    env = TowerEnv(TowerConfig(), render_mode="rgb_array")
    if args.scripted:
        script = ScriptedStack()
        act = lambda e, obs: script(e)  # noqa: E731
        act.reset = script.reset
    else:
        policy = load_sac(args.policy)
        act = lambda e, obs: binary_grip(e, policy(e, obs))  # noqa: E731
        act.reset = policy.reset
    rng = np.random.default_rng(args.seed)

    runs = []
    for i in range(args.rows * args.cols):
        sizes = MEASURED if args.sizes == "measured" else random_sizes(rng)
        frames, outcome = play(env, act, sizes, args.seed + i, args.every)
        runs.append((frames, outcome))
        print(f"seed {args.seed + i}: {outcome}, {len(frames)} frames", flush=True)
    block_rgb = [env.model.geom_rgba[env.gid[b]][:3] * 255 for b in BLOCKS]
    env.close()

    n = max(len(f) for f, _ in runs) + args.hold
    tiles = []
    for frames, outcome in runs:
        ok = outcome == "tower"
        text = "tower built" if ok else f"failed: {outcome}"
        lab = [label(f, text if k == len(frames) - 1 else "", ok, args.scale) for k, f in enumerate(frames)]
        tiles.append(lab + [lab[-1]] * (n - len(lab)))
    grid = []
    for k in range(n):
        rows = [np.concatenate([tiles[r * args.cols + c][k] for c in range(args.cols)], axis=1)
                for r in range(args.rows)]
        grid.append(np.concatenate(rows, axis=0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fps = round(1 / (TowerConfig().control_dt * args.every))
    # One palette for every frame, from a few spread frames plus swatches of the block and label colours, which
    # are too few pixels to survive the median cut: a fifth of imageio's size, no flicker.
    w = grid[0].shape[1]
    swatch = [np.full((40, w, 3), np.clip(np.asarray(c) * k, 0, 255), np.uint8)
              for c in [*block_rgb, (120, 230, 120), (240, 120, 120)] for k in (0.35, 0.55, 0.75, 1.0)]
    ref = Image.fromarray(np.concatenate([*grid[:: max(1, n // 6)], *swatch], axis=0)).quantize(
        128, Image.Quantize.MEDIANCUT)
    ims = [Image.fromarray(g).quantize(palette=ref, dither=Image.Dither.NONE) for g in grid]
    ims[0].save(args.out, save_all=True, append_images=ims[1:], duration=round(1000 / fps), loop=0, optimize=True)
    built = sum(o == "tower" for _, o in runs)
    print(f"{args.out}: {built}/{len(runs)} towers, {n} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
