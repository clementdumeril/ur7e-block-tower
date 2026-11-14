# ur7e-block-tower: reinforcement learning for block stacking on a UR7e, from MuJoCo to the real arm

![End-to-end SAC policy, no script at run time: four towers from random block poses and sizes](results/tower_sac.gif)

*One learned policy, no script at run time: four towers from random block orientations (8, 27, 55, 76 deg) and sizes.*

A MuJoCo replica of a lab UR7e cell (Robotiq Hand-E, wrist camera, wrist force sensing, the three measured lab blocks), used to train reinforcement learning to stack the blocks into a tower: black on green, then red on black.

- An end-to-end SAC policy, from block estimates to arm and gripper commands with no script at run time, builds 50 % of 100 simulated towers with random block sizes and orientations (first block 68 %), 33 % with the measured lab blocks.
- A residual PPO policy, a learned correction on top of a scripted stack, builds 83 % of 200 simulated towers under camera error. The scripted stack alone builds 91.5 %.
- The residual policy runs on the real UR7e through a ROS 2 node, with MoveIt IK and stops on contact force and tracking error.

[Results](docs/RESULTS.md) · [ROS 2 package for the real arm](ros2/tower_ros) · [Result files](results)

## How it works

- Perception: block poses come from a camera estimate with error (3 mm, 4 deg), replaced by a wrist-camera close-up (1 mm, 1.5 deg) once the gripper is above the block.
- Scripted stack ([`tower_scripted.py`](pensig/tower_scripted.py)): pick and carry, then a slow touchdown that stops on the wrist force.
- Residual RL ([`residual.py`](pensig/residual.py)): PPO (Stable-Baselines3) learns a bounded correction on the script's action, with a potential-based reward and the wrist force in the observation; damped least-squares IK at 20 Hz, and a protective stop on excess force, as on the real arm.
- End-to-end RL ([`train_tower_sac.py`](scripts/train_tower_sac.py), [`rlpd.py`](pensig/rlpd.py)): SAC with the script's episodes in half of every batch and a cloning term on them, episodes started from states saved along scripted runs, one phase earlier each time the deterministic policy succeeds from every start in use ([`curriculum.py`](pensig/curriculum.py)). [What it took](docs/RESULTS.md#end-to-end-sac).
- Real arm ([`ros2/tower_ros`](ros2/tower_ros)): the policy exported to numpy, the same 42 observations as in simulation, the wrist RealSense and force-torque sensor; `tower` runs the scripted stack alone.

## Run it

Python 3.10+, MuJoCo 3.2+. The UR5e model (same kinematics as the UR7e) is downloaded from MuJoCo Menagerie, not redistributed. The Python package is called `pensig`, its original name from the earlier pen task in this repository.

```bash
pip install -e ".[dev]" && python scripts/fetch_assets.py && python -m pytest -q
python scripts/evaluate_tower.py --scripted --episodes 200 --block-sizes 0.05 0.05 0.04 0.08 0.035 0.03
# the 83 % policy: residual PPO on the photo-estimated sizes, then 500k steps on the measured ones
python scripts/train.py --task tower --residual --timesteps 4000000 --out runs/tower_v5
python scripts/train.py --task tower --residual --block-sizes 0.05 0.05 0.04 0.08 0.035 0.03 --resume runs/tower_v5/final_model.zip --timesteps 500000 --out runs/tower_v6
python scripts/evaluate_tower.py --residual --policy runs/tower_v6/final_model.zip --episodes 200 --block-sizes 0.05 0.05 0.04 0.08 0.035 0.03
```

The end-to-end policy (32 cores, about 5 h to the best checkpoint):

```bash
python scripts/train_tower_sac.py --out runs/tower_sac --n-envs 32 --buffer 12000000 --max-minutes 780
python scripts/train_tower_sac.py --eval-only runs/tower_sac/best/best_model.zip --eval-towers 100
```

On the lab arm: [`ros2/tower_ros/README.md`](ros2/tower_ros/README.md).

## Limitations

- The residual policy (83 %) does not beat the scripted stack it corrects (91.5 %); its value so far is as a step toward a policy without the script.
- Success rates are from simulation; on the real arm, the scripted stack placed black on green and the policy's first runs reached the black block, with no real-arm success rate yet.
- The end-to-end policy is below the scripted stack (33 % vs 91.5 % on the measured blocks) and has not run on the real arm yet.
- The simulated arm is the UR5e model, a kinematic stand-in for the UR7e.

## Credits and license

MIT. [MuJoCo](https://github.com/google-deepmind/mujoco) and the UR5e model of [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3), downloaded by `scripts/fetch_assets.py`; [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for PPO; [Gymnasium](https://github.com/Farama-Foundation/Gymnasium). The cell geometry follows the UR7e cell of UC Berkeley's EECS 106A lab. The repository started as a pen grasp and signature task in the same cell (release [v0.1.0](https://github.com/clementdumeril/ur7e-block-tower/releases/tag/v0.1.0), [notes](docs/REPRODUCING.md)).
