"""Runs the scripted block tower (tower_ros.script) on the real UR7e with the lab cell's interfaces.

  MoveIt /compute_ik (group ur_manipulator, ik_link wrist_3_link, gripper down (0, 1, 0, 0) turned by
  the block yaw) and /plan_kinematic_path for the big moves; straight moves are IK waypoints sent as one
  trajectory. Both go to /scaled_joint_trajectory_controller/follow_joint_trajectory.
  Gripper: /toggle_gripper (std_srvs/Trigger) only, so the node keeps track of open/closed itself and
  assumes the gripper is open at startup.
  Contact: /force_torque_sensor_broadcaster/wrench, zeroed at rest before the descent. The touchdown is a
  dense trajectory at 1 cm/s that is cancelled as soon as the force rises 3 N above the zero, which
  stops the arm where it is (forward_velocity_controller would also do, but needs the Jacobian and a
  controller switch). Every other straight move aborts the run past guard_force.

  ros2 run tower_ros tower --ros-args --params-file <config>/tower.yaml -p step:=true

  After a stop (e-stop, guard force, failed trajectory) the node waits: release the e-stop, Enter, and it
  runs reset_state, opens the gripper if it was closed and starts again; stages already stacked are skipped.
  The gripper state is kept in gripper_state_file, so a new run after a crash knows it too.
"""

import math
import os
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import fields

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, WrenchStamped
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from tower_ros.geometry import (STAGES, ToolPose, combine_samples, contact_detected, estimate_from_marker,
                                linear_waypoints, tool_from_wrist, waypoint_times, wrist_from_tool)
from tower_ros.script import Robot, TowerAbort, TowerParams, TowerScript

UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def wait(future, timeout: float):
    """Result of a future the executor thread completes, or None after `timeout` s."""
    end = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > end:
            return None
        time.sleep(0.005)
    return future.result()


def joint_vector(js: JointState) -> np.ndarray:
    idx = {n: i for i, n in enumerate(js.name)}
    return np.array([js.position[idx[n]] for n in UR_JOINTS])


def duration(t: float) -> DurationMsg:
    return DurationMsg(sec=int(t), nanosec=int((t - int(t)) * 1e9))


class TowerRobot(Node, Robot):
    def __init__(self):
        super().__init__("tower")
        dp = self.declare_parameter
        self.step = bool(dp("step", True).value)  # wait for Enter before every motion and gripper toggle
        self.dry_run = bool(dp("dry_run", False).value)  # IK and log only, nothing moves
        self.stages = int(dp("stages", 2).value)  # 1: black on green only
        self.velocity_scaling = float(dp("velocity_scaling", 0.1).value)
        self.acceleration_scaling = float(dp("acceleration_scaling", 0.1).value)
        self.planner_id = dp("planner_id", "RRTConnectkConfigDefault").value
        self.avoid_collisions = bool(dp("avoid_collisions", True).value)
        self.linear_step = float(dp("linear_step", 0.005).value)
        self.touchdown_step = float(dp("touchdown_step", 0.001).value)
        self.max_joint_step = float(dp("max_joint_step", 0.15).value)  # rad between waypoints: no IK flips
        self.guard_force = float(dp("guard_force", 15.0).value)
        self.force_samples = int(dp("force_samples", 10).value)  # averaged for the contact test (~20 ms)
        self.settle = float(dp("settle", 1.0).value)  # s at rest before frames count for an estimate
        self.estimate_frames = int(dp("estimate_frames", 5).value)
        self.estimate_timeout = float(dp("estimate_timeout", 6.0).value)
        self.gripper_wait = float(dp("gripper_wait", 1.5).value)
        self.retry = bool(dp("retry", True).value)  # after a stop, wait for the operator and start again
        self.reset_command = dp("reset_command", "ros2 run ur7e_utils reset_state").value
        self.gripper_state_file = os.path.expanduser(dp("gripper_state_file", "~/.ros/tower_gripper_state").value)
        self.base_frame = dp("base_frame", "base_link").value
        self.wrist_frame = dp("wrist_frame", "wrist_3_link").value
        self.group = dp("group", "ur_manipulator").value

        defaults = TowerParams()
        values = {}
        for f in fields(TowerParams):
            if f.name == "block_heights":
                continue
            v = getattr(defaults, f.name)
            got = dp(f.name, list(v) if isinstance(v, tuple) else v).value
            values[f.name] = tuple(got) if isinstance(v, tuple) else type(v)(got)
        values["block_heights"] = {c: float(dp(f"{c}_height", h).value) for c, h in defaults.block_heights.items()}
        self.tower_params = TowerParams(**values)

        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10, callback_group=cb)
        self.create_subscription(WrenchStamped, dp("wrench_topic", "/force_torque_sensor_broadcaster/wrench").value,
                                 self.wrench_cb, 50, callback_group=cb)
        self.create_subscription(MarkerArray, "/tower/blocks", self.blocks_cb, 10, callback_group=cb)
        self.target_pub = self.create_publisher(PoseStamped, "/tower/target", 1)
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik", callback_group=cb)
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path", callback_group=cb)
        self.gripper_cli = self.create_client(Trigger, dp("gripper_service", "/toggle_gripper").value,
                                              callback_group=cb)
        self.zero_cli = self.create_client(Trigger, dp("zero_service", "/io_and_status_controller/zero_ftsensor").value,
                                           callback_group=cb)
        self.switch_cli = self.create_client(SwitchController, "/controller_manager/switch_controller",
                                             callback_group=cb)
        self.exec_ac = ActionClient(self, FollowJointTrajectory,
                                    "/scaled_joint_trajectory_controller/follow_joint_trajectory", callback_group=cb)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lock = threading.Lock()
        self.joints = None
        self.start_joints = None
        self.forces = deque(maxlen=2000)  # (monotonic time, force xyz)
        self.block_msgs = deque(maxlen=200)  # (stamp ns, {color: BlockEstimate})
        self.zero = None
        # /toggle_gripper only toggles: the state is tracked here and kept across runs in gripper_state_file
        self.closed = self._read_gripper_state()
        self.active_goal = None
        self.dry_seed = None
        self.dry_pose = None

    # ------------------------------------------------------------ callbacks

    def joint_cb(self, msg: JointState):
        if not set(UR_JOINTS) <= set(msg.name):
            return
        self.joints = msg
        if self.start_joints is None:
            self.start_joints = msg
            self.get_logger().info("Recorded the start pose, each stage starts and estimates from here")

    def wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        with self.lock:
            self.forces.append((time.monotonic(), np.array([f.x, f.y, f.z])))

    def blocks_cb(self, msg: MarkerArray):
        ests, stamp = {}, None
        for m in msg.markers:
            if m.action != Marker.ADD:
                continue
            p, q = m.pose.position, m.pose.orientation
            n = int(m.text) if m.text.isdigit() else 0
            ests[m.ns] = estimate_from_marker((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), m.scale.z, n)
            stamp = Time.from_msg(m.header.stamp).nanoseconds
        if stamp is not None:
            with self.lock:
                self.block_msgs.append((stamp, ests))

    # ------------------------------------------------------------ setup

    def wait_ready(self):
        end = time.monotonic() + 10.0
        while self.start_joints is None:
            if time.monotonic() > end:
                raise TowerAbort("no /joint_states")
            time.sleep(0.05)
        for cli, name in ((self.ik_cli, "/compute_ik"), (self.plan_cli, "/plan_kinematic_path")):
            if not cli.wait_for_service(timeout_sec=10.0):
                raise TowerAbort(f"no {name}: is MoveIt up?")
        if self.dry_run:
            return
        time.sleep(0.5)
        if self._recent_forces() is None:
            raise TowerAbort("no wrench on the wrench topic: the touchdown needs it")

    def switch_to_trajectory_controller(self):
        """After free drive the trajectory controller may be inactive."""
        if not self.switch_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("No controller_manager, assuming scaled_joint_trajectory_controller is active")
            return
        req = SwitchController.Request()
        req.activate_controllers = ["scaled_joint_trajectory_controller"]
        req.deactivate_controllers = ["freedrive_mode_controller", "forward_position_controller",
                                      "forward_velocity_controller"]
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        res = wait(self.switch_cli.call_async(req), 5.0)
        if res is None or not res.ok:
            self.get_logger().warn("Could not switch to scaled_joint_trajectory_controller")

    # ------------------------------------------------------------ Robot interface

    def go_start(self, label: str):
        if self.dry_run:
            self.get_logger().info(f"[dry run] {label}")
            self.dry_seed, self.dry_pose = self.start_joints, None
            return
        traj = self._plan(self.start_joints)
        self._confirm(f"{label} (planned, {self._traj_time(traj):.1f} s)")
        self._execute(traj, None)

    def estimate(self, colors, close: bool) -> dict:
        t0 = self.get_clock().now().nanoseconds + int(self.settle * 1e9)
        end = time.monotonic() + self.settle + self.estimate_timeout
        samples = {c: [] for c in colors}
        used = set()
        while time.monotonic() < end and any(len(s) < self.estimate_frames for s in samples.values()):
            time.sleep(0.05)
            with self.lock:
                msgs = list(self.block_msgs)
            for stamp, ests in msgs:
                if stamp < t0 or stamp in used:
                    continue
                used.add(stamp)
                for c in colors:
                    if c in ests and len(samples[c]) < self.estimate_frames:
                        samples[c].append(ests[c])
        out = {c: combine_samples(s) for c, s in samples.items() if len(s) >= max(1, self.estimate_frames // 2)}
        kind = "close-up" if close else "coarse"
        for c in colors:
            if c in out:
                e = out[c]
                self.get_logger().info(f"{kind} {c}: ({e.x:.3f}, {e.y:.3f}) top {e.top:.3f} "
                                       f"height {e.height * 1000:.0f} mm yaw {math.degrees(e.yaw):.1f} deg "
                                       f"({len(samples[c])} frames)")
            else:
                self.get_logger().warn(f"{kind}: no {c} block ({len(samples[c])} frames)")
        return out

    def move(self, pose: ToolPose, label: str, speed=None, stop_force=None) -> bool:
        self._publish_target(pose)
        desc = (f"{label}: fingertips ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}) yaw {math.degrees(pose.yaw):.1f} deg"
                + (" planned" if speed is None else f" straight at {speed * 100:.1f} cm/s")
                + (f", stop at {stop_force:.0f} N" if stop_force else ""))
        seed = self._seed()
        if speed is None:
            sol = self._ik(pose, seed)
            if sol is None:
                raise TowerAbort(f"no IK for {desc}")
            if self.dry_run:
                return self._dry(desc, pose, sol)
            traj = self._plan(sol)
            self._confirm(f"{desc} ({self._traj_time(traj):.1f} s)")
            return self._execute(traj, None)

        start = self._current_tool()
        wps = linear_waypoints(start, pose, self.touchdown_step if stop_force else self.linear_step)
        times = waypoint_times(start, wps, speed)
        prev, qs = joint_vector(seed), []
        for wp in wps:
            sol = self._ik(wp, seed)
            if sol is None:
                raise TowerAbort(f"no IK at ({wp.x:.3f}, {wp.y:.3f}, {wp.z:.3f}) on the way: {desc}")
            q = joint_vector(sol)
            if np.max(np.abs(q - prev)) > self.max_joint_step:
                raise TowerAbort(f"IK jumps {np.max(np.abs(q - prev)):.2f} rad on the way: {desc}")
            qs.append(q)
            prev, seed = q, sol
        if self.dry_run:
            return self._dry(f"{desc}, {len(wps)} waypoints, {times[-1]:.1f} s", pose, seed)
        traj = JointTrajectory(joint_names=list(UR_JOINTS), points=[
            JointTrajectoryPoint(positions=[float(v) for v in q], time_from_start=duration(t))
            for q, t in zip(qs, times)])
        if stop_force and self.zero is None:
            raise TowerAbort("contact move without a zeroed wrench")
        self._confirm(f"{desc} ({times[-1]:.1f} s)")
        return self._execute(traj, stop_force)

    def zero_wrench(self):
        if self.dry_run:
            self.get_logger().info("[dry run] zero the wrench")
            return
        if self.zero_cli.wait_for_service(timeout_sec=1.0):
            res = wait(self.zero_cli.call_async(Trigger.Request()), 3.0)
            if res is None or not res.success:
                self.get_logger().warn("zero_ftsensor failed, using the software zero only")
        time.sleep(0.5)
        forces = self._recent_forces(n=150)
        if forces is None:
            raise TowerAbort("no wrench to zero")
        self.zero = forces.mean(0)

    def gripper(self, close: bool, label: str):
        if close == self.closed:
            self.get_logger().warn(f"{label}: the gripper should already be {'closed' if close else 'open'}, skipping")
            return
        if self.dry_run:
            self.get_logger().info(f"[dry run] toggle gripper: {label}")
            self.closed = close
            return
        self._confirm(f"toggle gripper: {label}")
        if not self.gripper_cli.wait_for_service(timeout_sec=5.0):
            raise TowerAbort("no /toggle_gripper service")
        res = wait(self.gripper_cli.call_async(Trigger.Request()), 5.0)
        if res is None:
            raise TowerAbort("/toggle_gripper did not answer")
        if not res.success:
            self.get_logger().warn(f"/toggle_gripper: {res.message}")
        self.closed = close
        self._write_gripper_state()
        time.sleep(self.gripper_wait)

    # ------------------------------------------------------------ recovery

    def _read_gripper_state(self) -> bool:
        try:
            with open(self.gripper_state_file) as f:
                return f.read().strip() == "closed"
        except OSError:
            return False

    def _write_gripper_state(self):
        try:
            os.makedirs(os.path.dirname(self.gripper_state_file), exist_ok=True)
            with open(self.gripper_state_file, "w") as f:
                f.write("closed" if self.closed else "open")
        except OSError as e:
            self.get_logger().warn(f"Could not save the gripper state: {e}")

    def open_if_closed(self):
        if self.closed:
            self.get_logger().warn(f"The gripper was left closed ({self.gripper_state_file}): opening it. If it is "
                                   "already open, stop here and toggle it once by hand.")
            self.gripper(False, "open before starting")

    def recover(self) -> bool:
        """After a stop: wait for the operator, reset the arm, and say whether to start again."""
        answer = input("\n[stopped] Fix the cause, release the e-stop (turn it clockwise), then Enter to reset the "
                       "arm and resume (stacked stages are skipped), q to quit: ")
        if answer.strip().lower().startswith("q"):
            return False
        self.get_logger().info(f"Running: {self.reset_command}")
        try:
            res = subprocess.run(shlex.split(self.reset_command), timeout=60)
            if res.returncode != 0:
                self.get_logger().warn(f"{self.reset_command} exited with {res.returncode}")
        except (OSError, subprocess.TimeoutExpired) as e:
            self.get_logger().warn(f"{self.reset_command} failed ({e}): run it by hand")
            input("Enter once the arm is reset: ")
        time.sleep(2.0)
        self.switch_to_trajectory_controller()
        self.zero = None
        self.open_if_closed()
        return True

    # ------------------------------------------------------------ helpers

    def _confirm(self, desc: str):
        if not self.step:
            self.get_logger().info(desc)
            return
        answer = input(f"\n[step] {desc}\n       Enter to run, q to stop: ")
        if answer.strip().lower().startswith("q"):
            raise TowerAbort("stopped by the operator")

    def _dry(self, desc: str, pose: ToolPose, sol: JointState) -> bool:
        self.get_logger().info(f"[dry run] {desc}")
        self.dry_seed, self.dry_pose = sol, pose
        return False

    def _seed(self) -> JointState:
        if self.dry_run and self.dry_seed is not None:
            return self.dry_seed
        return self.joints

    def _current_tool(self) -> ToolPose:
        if self.dry_run and self.dry_pose is not None:
            return self.dry_pose
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.wrist_frame, Time())
        except TransformException as ex:
            raise TowerAbort(f"no TF {self.base_frame} -> {self.wrist_frame}: {ex}")
        t, q = tf.transform.translation, tf.transform.rotation
        return tool_from_wrist((t.x, t.y, t.z), (q.x, q.y, q.z, q.w), self.tower_params.gripper_length)

    def _publish_target(self, pose: ToolPose):
        xyz, q = wrist_from_tool(pose, self.tower_params.gripper_length)
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in xyz)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q
        self.target_pub.publish(msg)

    def _ik(self, pose: ToolPose, seed: JointState):
        xyz, q = wrist_from_tool(pose, self.tower_params.gripper_length)
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.group
        ik.ik_link_name = self.wrist_frame
        ik.avoid_collisions = self.avoid_collisions
        ik.timeout = DurationMsg(sec=1)
        ik.robot_state.joint_state = seed
        ik.pose_stamped.header.frame_id = self.base_frame
        ik.pose_stamped.pose.position.x, ik.pose_stamped.pose.position.y, ik.pose_stamped.pose.position.z = (
            float(v) for v in xyz)
        o = ik.pose_stamped.pose.orientation
        o.x, o.y, o.z, o.w = (float(v) for v in q)
        res = wait(self.ik_cli.call_async(req), 5.0)
        if res is None or res.error_code.val != res.error_code.SUCCESS:
            return None
        return res.solution.joint_state

    def _plan(self, target: JointState) -> JointTrajectory:
        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = self.group
        mpr.allowed_planning_time = 5.0
        mpr.num_planning_attempts = 3
        mpr.planner_id = self.planner_id
        mpr.max_velocity_scaling_factor = self.velocity_scaling
        mpr.max_acceleration_scaling_factor = self.acceleration_scaling
        goal = Constraints()
        for name, pos in zip(UR_JOINTS, joint_vector(target)):
            goal.joint_constraints.append(JointConstraint(joint_name=name, position=float(pos), tolerance_above=0.01,
                                                          tolerance_below=0.01, weight=1.0))
        mpr.goal_constraints.append(goal)
        res = wait(self.plan_cli.call_async(req), 15.0)
        if res is None or res.motion_plan_response.error_code.val != 1:
            raise TowerAbort("MoveIt found no plan")
        return res.motion_plan_response.trajectory.joint_trajectory

    @staticmethod
    def _traj_time(traj: JointTrajectory) -> float:
        if not traj.points:
            return 0.0
        d = traj.points[-1].time_from_start
        return d.sec + d.nanosec * 1e-9

    def _recent_forces(self, n=None):
        n = n or self.force_samples
        with self.lock:
            if not self.forces or time.monotonic() - self.forces[-1][0] > 0.2:
                return None
            return np.array([f for _, f in list(self.forces)[-n:]])

    def _execute(self, traj: JointTrajectory, stop_force) -> bool:
        """Runs a trajectory; stops it on contact when stop_force is set, aborts past guard_force otherwise."""
        if not self.exec_ac.wait_for_server(timeout_sec=5.0):
            raise TowerAbort("no trajectory controller action server")
        forces = self._recent_forces()
        if forces is None:
            raise TowerAbort("the wrench topic stopped")
        guard_zero = forces.mean(0)
        gh = wait(self.exec_ac.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)), 5.0)
        if gh is None or not gh.accepted:
            raise TowerAbort("the trajectory controller rejected the goal")
        self.active_goal = gh
        result = gh.get_result_async()
        touched = False
        try:
            while not result.done():
                forces = self._recent_forces()
                if forces is None:
                    raise TowerAbort("the wrench topic stopped during a move")
                if stop_force and contact_detected(forces, self.zero, stop_force):
                    touched = True
                    break
                if contact_detected(forces, guard_zero, self.guard_force):
                    raise TowerAbort(f"more than {self.guard_force:.0f} N on the wrist")
                time.sleep(0.002)
        except TowerAbort:
            self.stop_motion()
            raise
        if touched:
            f = np.linalg.norm(forces.mean(0) - self.zero)
            self.stop_motion()
            wait(result, 5.0)
            time.sleep(0.3)
            z = self._current_tool().z
            self.get_logger().info(f"contact: {f:.1f} N, fingertips stopped at z {z:.4f}")
            return True
        self.active_goal = None
        res = result.result()
        if res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise TowerAbort(f"trajectory failed ({res.result.error_code}): {res.result.error_string}")
        return False

    def stop_motion(self):
        """Cancel the running trajectory: the controller holds the arm where it is."""
        gh, self.active_goal = self.active_goal, None
        if gh is not None:
            wait(gh.cancel_goal_async(), 2.0)


def main(args=None):
    rclpy.init(args=args)
    node = TowerRobot()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    log = node.get_logger()
    try:
        node.wait_ready()
        mode = "dry run" if node.dry_run else ("step mode" if node.step else "CONTINUOUS")
        log.info(f"Tower on the real arm, {mode}, gripper_length {node.tower_params.gripper_length:.3f} m. "
                 "The gripper must be open now.")
        if not node.dry_run:
            node.switch_to_trajectory_controller()
            node.open_if_closed()
        stages = STAGES[:max(1, min(node.stages, len(STAGES)))]
        while True:
            try:
                TowerScript(node, node.tower_params, log=log.info).run(stages)
                log.info("Tower done")
                break
            except TowerAbort as e:
                node.stop_motion()
                log.error(f"Stopped: {e}")
                if str(e) == "stopped by the operator" or node.dry_run or not node.retry or not node.recover():
                    break
                log.info("Starting again from the start pose")
    except TowerAbort as e:
        log.error(f"Stopped: {e}")
    except KeyboardInterrupt:
        log.warn("Interrupted")
    finally:
        try:
            node.stop_motion()
        except Exception as e:  # the context may already be down after Ctrl-C: the e-stop is the real stop
            log.error(f"Could not cancel the trajectory: {e}")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
