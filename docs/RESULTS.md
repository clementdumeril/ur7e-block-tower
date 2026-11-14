# Results

Back to the [README](../README.md).

## Method

- Perception. Block poses come from a camera estimate with error (3 mm, 4 deg), replaced by a wrist-camera close-up (1 mm, 1.5 deg) once the gripper is above the block. The green block is wider than the 50 mm Hand-E stroke, so it stays in place as the base.
- Scripted stack ([`tower_scripted.py`](../pensig/tower_scripted.py)): pick and carry, going down only on the wrist estimate, then a touchdown at 6 cm/s and 1 cm/s over the last 12 mm that stops on the wrist force, like `moveUntilContact`.
- Learning ([`residual.py`](../pensig/residual.py), `ResidualTowerEnv`): PPO (Stable-Baselines3) learns a bounded correction on top of the script, with a potential-based reward and the wrist force in the observation. Damped least-squares IK turns the TCP target into joint commands at 20 Hz, and a protective stop ends the episode when the gripper or a block presses too hard, as on the real arm.
- Real arm ([`ros2/tower_ros`](../ros2/tower_ros)): `tower_policy` runs the PPO policy exported to numpy on the UR7e, with the same 42 observations as in simulation, MoveIt IK, the wrist RealSense and the wrist force-torque sensor. It waits for the arm to reach each command and stops on a tracking error or a contact force above the limit. `tower` runs the scripted stack alone.

## Residual PPO and scripted stack

200 full towers under camera error. A placement counts within 8 mm and 10 deg, released and cleared. Placement error is the mean of the two stages.

Measured lab blocks (green 50 mm, black 80 mm, red 30 mm high):

| | Full tower | Black on green | Red on black | Placement error | Source |
|---|---|---|---|---|---|
| Scripted stack | 91.5 % | 93.5 % | 97.9 % | 1.6 mm | [`tower_scripted_measured.json`](../results/tower_scripted_measured.json) |
| PPO residual (`tower_v6`: `tower_v5` fine-tuned 500k steps on these sizes) | 83 % | 90.5 % | 91.7 % | 2.0 mm | [`tower_residual_measured.json`](../results/tower_residual_measured.json) |

Block sizes estimated from a photo, and what the wrist close-up brings:

| | Full tower | Black on green | Red on black | Placement error | Source |
|---|---|---|---|---|---|
| Scripted stack | 93 % | 96 % | 96.9 % | 1.6 mm | [`tower_scripted_noisy.json`](../results/tower_scripted_noisy.json) |
| PPO residual (`tower_v5`, trained on these sizes) | 78 % | 86 % | 90.7 % | 2.0 mm | [`tower_residual_noisy.json`](../results/tower_residual_noisy.json) |
| Scripted stack, first estimate only | 52 % | 66 % | 78.8 % | 4.0 mm | [`tower_scripted_nowrist.json`](../results/tower_scripted_nowrist.json) |
| Scripted stack, exact poses | 98.5 % | 98.5 % | 100 % | 0.3 mm | [`tower_scripted_noiseless.json`](../results/tower_scripted_noiseless.json) |

`tower_v5` evaluated on the measured blocks before fine-tuning did worse than after it; that evaluation file was not committed, so its number is not given here.

The gain came from information, not training: without the close-up, most failures were a finger landing on top of the black block or a block set down off center, from a camera error the policy could not see.

## Real arm

On the real UR7e, the scripted stack placed black on green on its first run in step mode, stopping at 3.1 N on contact. The first policy runs on the arm reached the black block; one stopped at 74 N when the top of the 80 mm block met the Hand-E palm, fixed by grasping 10 mm higher (`grasp_raise`). These are run notes; no real-arm success rate is measured yet.

## End-to-end PPO, no script at run time

From the plain reset, PPO never finds a placement, so it never sees a success to learn from. [`train_tower_curriculum.py`](../scripts/train_tower_curriculum.py) starts episodes partway through the task, from states saved along scripted runs, and moves the start one phase earlier once the policy succeeds from there ([`curriculum.py`](../pensig/curriculum.py); start states from demonstrations as in Nair et al. 2018 and Salimans and Chen 2018, reverse curriculum as in Florensa et al. 2017). Block sizes are random at every reset.

First run on the Savio cluster, 26M steps in 11 h, 50 episodes from each start phase (from the run's evaluation log, not committed):

| Start phase | 7 back up | 6 open | 5 touchdown | 4 over the tower | 3 lift | 0-2 grasp |
|---|---|---|---|---|---|---|
| Success | 90 % | 76 % | 74 % | 6 % | 0 % | 0 % |

The policy learned the end of the task alone and passed the carry and lift phases at 5M steps, then stalled on the grasp and forgot them: most failures from the earlier phases are timeouts, with the arm hovering, and protective stops against the table or the tower. PPO never got further; SAC below did.

## End-to-end SAC

Why: the residual policy needs the script at run time. The goal is one network that picks and stacks alone.

How ([`train_tower_sac.py`](../scripts/train_tower_sac.py), [`rlpd.py`](../pensig/rlpd.py)):

- SAC, off-policy, so failures and 2000 scripted episodes are reused: half of every batch comes from the script (RLPD, Ball et al. 2023), with a cloning term on that half (TD3+BC, Fujimoto and Gu 2021). The actor starts as a copy of the script.
- Start-state curriculum as above, gated on the deterministic policy, the one the arm runs.
- 29 observations: TCP pose, block estimates relative to the TCP, wrist force, gripper, previous action. No script phase, no clock.

Result, 100 towers per checkpoint, deterministic policy, camera error on ([`tower_sac_eval.json`](../results/tower_sac_eval.json)):

| Checkpoint | Full tower, random blocks | First block | Full tower, measured blocks |
|---|---|---|---|
| 2M steps | 47 % | 63 % | 36 % |
| 4M steps | **50 %** | 68 % | 33 % |
| 7M steps | 9 % | 48 % | 8 % |

What it took. Each failed run was the policy reading a shortcut in its observation:

- Elapsed time in the observation: it closed the gripper on the clock, not on the block. Removed.
- Open above the tower looked the same before and after a placement: it closed on nothing, flew to the tower and hovered. Added a flag set when the gripper closes on the block.
- The wrist close-up of the tower was only ever taken after a carry, so the policy used it as a phase cue and failed whenever it started over the tower, as after its own first placement. Half the episodes now start there; full towers went from 0-7 % to 50 %.
- Gripper noise: the mean action stayed closed while sampled actions opened by chance. Gripper std capped, gripper left out of the cloning term.
- After 4M steps the replay buffer (3M) wrapped, the critic loss doubled and towers fell to 9 %. Buffer now 12M, and the best checkpoint is kept.
- Most failures on the measured blocks were protective stops: grasping 15 mm over the bottom put the palm into the 80 mm black block. The grasp now rises on tall blocks; the next run trains with it.

## What the simulation taught me

- Paying the hold over the tower as much as the release taught the first policy to keep the block in hand.
- Paid per step, holding the block was worth more than trying to place it (success bonus 20 steps later, gamma 0.98). Paying progress as a potential difference stopped the drift.
- An initial exploration std of 0.3 took the script from 50 % to 20 % of placements; the tower starts at 0.1.
- A released block dropping 2 mm spiked the contact force past the protective stop: the held block counts only while a finger touches it.
- Coming down at full speed, the arm lagged the command by 9 mm and landed hard; the touchdown starts at 6 cm/s.
- The 1 kg gripper accelerating shows on the wrist force as much as a block touching down: the touchdown check only runs over the last 12 mm, at constant speed.
- A policy trained on the photo sizes did worse on the measured blocks; 500k steps of fine-tuning brought it to 83 %.

## Code structure

| Path | Role |
|---|---|
| [`scene.py`](../pensig/scene.py) | UR5e from Menagerie, Hand-E with the lab fingers (fingertips 16 cm below the flange), wrist camera, lab pedestal |
| [`tower.py`](../pensig/tower.py) | Gymnasium env: blocks, camera and wrist estimates, wrist force, reward, protective stop |
| [`tower_scripted.py`](../pensig/tower_scripted.py) | scripted pick, carry and force-stopped touchdown |
| [`ik.py`](../pensig/ik.py), [`residual.py`](../pensig/residual.py) | damped least-squares IK; env where PPO learns a correction on the script |
| [`curriculum.py`](../pensig/curriculum.py) | episodes started partway by the script, for the end-to-end policy |
| [`ros2/tower_ros`](../ros2/tower_ros) | perception, scripted tower and policy nodes for the real UR7e |
| [`scripts/`](../scripts) | train, evaluate, export the policy, world-model planning, cluster jobs |

## Earlier task in this repository

The repository started with a pen grasp and signature in the same cell (release [v0.1.0](https://github.com/clementdumeril/ur7e-block-tower/releases/tag/v0.1.0)). That code is still in `pensig/env.py`, `scripted.py` and `draw.py`, which is why the package is called `pensig`; see [REPRODUCING.md](REPRODUCING.md) and [SIMULATION_NOTES.md](SIMULATION_NOTES.md).
