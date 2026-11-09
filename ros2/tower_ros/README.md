# tower_ros

The scripted block tower from `pensig/tower_scripted.py` on the lab UR7e: black on green, then red on black.
It uses the lab cell's interfaces (MoveIt IK and planning, `/toggle_gripper`, the wrist RealSense) and adds a
close-up estimate from the wrist camera before each descent and a touchdown that stops on the wrist force.

## Build

In the lab workspace (the collision boxes of its `planning` package are loaded if present):

```
ln -s ~/ur7e-block-tower/ros2/tower_ros ~/<workspace>/src/tower_ros
cd ~/<workspace> && colcon build --packages-select tower_ros && source install/setup.bash
```

## Run

UR driver up, then:

```
ros2 launch tower_ros tower_bringup.launch.py
ros2 run tower_ros tower --ros-args --params-file install/tower_ros/share/tower_ros/config/tower.yaml
```

The arm's pose when `tower` starts is the start pose: each stage estimates from there. Add `-p dry_run:=true`
to only log targets and check IK, `-p stages:=1` for black on green only, `-p step:=false` once step mode has
gone through cleanly. RViz: `/tower/blocks`, `/tower/block_points`, `/tower/target`.

Continuous and about twice as fast, for a video (the touchdown stays at 1 cm/s):

```
ros2 run tower_ros tower --ros-args --params-file install/tower_ros/share/tower_ros/config/tower.yaml --params-file install/tower_ros/share/tower_ros/config/tower_fast.yaml
```

After an e-stop or any stop, the node waits: release the e-stop, press Enter, and it runs `reset_state`,
opens the gripper if it was left closed and starts again from the start pose. A stage already stacked is
skipped, so the tower goes on where it stopped. The gripper state is kept in `~/.ros/tower_gripper_state`.

## Safety

- Step mode first (the default): every motion and gripper toggle waits for Enter. Keep a hand on the e-stop and
  the pendant speed slider low.
- Measure first: `gripper_length` (flange to fingertips: 0.16 measured on the lab Hand-E; too short sends
  the fingers into the table), the block heights, and the black top against `/tower/black/top` (the
  RealSense reads dark tops low; set `black_top_offset`).
- `/toggle_gripper` only toggles: the node opens the gripper at startup if its last run left it closed. If it
  is open while the file says closed, toggle it once by hand first.
- The lab workspace's gripper collision box reaches 18.5 cm below the flange: if IK fails near the table, check
  `obstacles.yaml`.

## Policy

`tower_policy` runs the residual PPO policy (`runs/tower_v6`, trained on the measured blocks) instead of
the script: pensig's ScriptedStack action plus the network's correction, one MoveIt IK point per period on
the trajectory controller. In sim it builds 83 % of towers against 91.5 % for the script, so this is an
experiment; `tower` stays the reliable way.

Once, on the dev machine, export the network to numpy (no torch or MuJoCo needed on the arm), then copy
`policy.npz` and `config.json` to `runs/tower_v6/` in the checkout on the lab machine (`repo`; when empty,
`~/ur7e-block-tower`, else `~/ur-pen-signature`, the repository's earlier name):

```
python scripts/export_tower_policy.py runs/tower_v6
```

Run, dry run first (IK every step, the fingertips assumed to follow the command and to touch down on the
estimated top, nothing moves):

```
ros2 run tower_ros tower_policy --ros-args --params-file install/tower_ros/share/tower_ros/config/tower.yaml -p dry_run:=true
```

- Each stage: coarse estimate from the start pose, close-ups from 0.30 m (the camera is 13 cm off the
  gripper and sees nothing closer), a start over the held block as in the sim, then the loop. The policy
  gets each close-up when the TCP reaches the sim's wrist trigger.
- `time_scale: 3` plays the sim's motion at a third of its speed (up to 0.17 m/s). Slow down with
  `time_scale` only: `max_speed` and `max_yaw_rate` are backstops, and a binding cap changes what the policy
  sees (in a dry run at 0.1 m/s it stalled the second stage, its yaw correction holding the script's
  alignment check open). Keep the pendant slider at 100 %: lower, the arm lags and the tracking guard stops it.
- It stops and holds on: more than 15 N, 160 steps in a stage, fingertips 4 cm off the command, the robot
  program stopping (e-stop), Ctrl+C. After an e-stop: `ros2 run ur7e_utils reset_state`.
- Check `force_sign` once: push the fingers up during a dry run, the logged `f` z must go positive.

`colcon test` or `python -m pytest ros2/tower_ros` runs the geometry, phase and policy tests without ROS.
