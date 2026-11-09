"""Runs the residual PPO tower policy (runs/tower_v6) on the real UR7e instead of the step-by-step script.

Per stage, outside the control loop (MoveIt-planned moves, step mode asks before each):
  start pose      coarse estimate of all blocks from /tower/blocks, table height from the first one
  close-ups       wrist camera 0.30 m above the held block, then above the target, averaged estimates
  sim start       TCP over the held block, 12 cm above the highest block, as TowerEnv starts a stage
Then the loop, one period per control_dt * time_scale: the 29 TowerEnv values, ScriptedStack's action and
phase (pensig's own script, reading a StageState), the network, total = clip(script + scale * residual),
target TCP += a[:3] * 0.015 m and yaw += a[3] * 0.15 rad (speed capped, clipped to the sim workspace and
a base_link box), MoveIt /compute_ik seeded with the last command, one point on
/scaled_joint_trajectory_controller/joint_trajectory reached at the end of the period.

Why this interface: the trajectory controller is the one MoveIt and tower_node already use (no controller
switch), it follows the pendant speed slider, and if this node dies the arm stops at the last point;
forward_velocity_controller would keep the last velocity and needs the Jacobian. /compute_ik takes a few ms
against a 50 ms period and checks the cell's collision boxes.

The wrist camera sits 13 cm off the gripper axis and the RealSense sees nothing closer than ~0.28 m, so it
cannot look at a block from where the sim's trigger fires (TCP within 1 cm of the block, < 25 cm above).
The close-ups are taken before the loop from 0.30 m and only revealed to the policy when the sim trigger
fires; the blocks do not move in between.

  ros2 run tower_ros tower_policy --ros-args --params-file <config>/tower.yaml -p dry_run:=true
"""

import math
import threading
import time
from collections import deque
from pathlib import Path

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
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from tower_ros.geometry import (BLOCKS, STAGES, ToolPose, combine_samples, estimate_from_marker, quat_to_matrix,
                                tool_for_camera_above, tool_from_wrist, wrist_from_tool)
from tower_ros.policy import (RESIDUAL_SCALE, GripperModel, PolicyFrame, SimConfig, StageState, closing_on_block,
                              load_policy, residual_obs, resolve_repo, scripted_stack_class, sensor_force, sim_start,
                              stack_below, stage_done, step_target, total_action)

UR_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


class PolicyAbort(RuntimeError):
    """Stop where the arm is and leave it to the operator."""


def wait(future, timeout: float):
    end = time.monotonic() + timeout
    while not future.done():
        if time.monotonic() > end:
            return None
        time.sleep(0.001)
    return future.result()


def joint_vector(js: JointState) -> np.ndarray:
    idx = {n: i for i, n in enumerate(js.name)}
    return np.array([js.position[idx[n]] for n in UR_JOINTS])


def joint_state(q) -> JointState:
    return JointState(name=list(UR_JOINTS), position=[float(v) for v in q])


def duration(t: float) -> DurationMsg:
    return DurationMsg(sec=int(t), nanosec=int((t - int(t)) * 1e9))


class TowerPolicy(Node):
    def __init__(self):
        super().__init__("tower_policy")
        dp = self.declare_parameter
        self.repo = resolve_repo(dp("repo", "").value)
        policy = Path(dp("policy", "runs/tower_v6/policy.npz").value).expanduser()
        self.policy_path = policy if policy.is_absolute() else self.repo / policy
        self.dry_run = bool(dp("dry_run", False).value)  # IK and log, nothing moves
        self.step = bool(dp("step", True).value)  # Enter before each planned move and before each loop
        self.stages = int(dp("stages", 2).value)
        self.time_scale = float(dp("time_scale", 3.0).value)  # real period = control_dt * time_scale
        # Backstops on the TCP target step, above what the sim commands at time_scale 3 (0.5 m/s / 3, 0.15 rad
        # per step). A binding cap changes the trajectory the policy sees: slow down with time_scale instead.
        self.max_speed = float(dp("max_speed", 0.20).value)  # m/s
        self.max_yaw_rate = float(dp("max_yaw_rate", 1.0).value)  # rad/s
        self.lookahead = float(dp("lookahead", 1.0).value)  # periods until the streamed point is reached
        self.step_limit = int(dp("step_limit", 160).value)  # sim episode_steps
        self.guard_force = float(dp("guard_force", 15.0).value)
        self.force_sign = float(dp("force_sign", -1.0).value)
        self.force_samples = int(dp("force_samples", 10).value)
        self.tracking_tol = float(dp("tracking_tol", 0.04).value)  # m between command and fingertips
        # After each period, wait until the fingertips are this close to the command (at most arrive_timeout):
        # the sim's servo reaches its target every period, and a slowed controller (pendant slider, reduced
        # mode) then only stretches time instead of letting the command run ahead of the arm.
        self.arrive_tol = float(dp("arrive_tol", 0.003).value)
        self.arrive_timeout = float(dp("arrive_timeout", 3.0).value)
        self.max_joint_step = float(dp("max_joint_step", 0.2).value)  # rad per period: no IK branch flips
        self.gripper_threshold = float(dp("gripper_threshold", 0.5).value)
        self.gripper_wait = float(dp("gripper_wait", 1.0).value)  # s the loop pauses for a toggle
        # Grasp this much higher on the held block than the sim (finger_above_bottom 15 mm), and so release it
        # this much higher over the tower: with 15 mm the top of the 80 mm black block met the inside of the
        # real Hand-E. Both estimates move up together, so the policy sees the same geometry.
        self.grasp_raise = float(dp("grasp_raise", 0.01).value)
        self.closeup = bool(dp("closeup", True).value)
        self.closeup_height = float(dp("closeup_height", 0.30).value)
        self.camera_offset = tuple(dp("camera_offset", [-0.025, 0.13, 0.0]).value)
        self.gripper_length = float(dp("gripper_length", 0.16).value)
        self.finger_axis_yaw = float(dp("finger_axis_yaw", 0.0).value)
        self.world_yaw = float(dp("world_yaw", 0.0).value)
        # true: shift xy so the blocks' center lands on the center of the training area (the sim clips the
        # target to |y| < 0.35 m in its frame); false: world_offset as given
        self.auto_offset = bool(dp("auto_offset", True).value)
        self.world_offset = tuple(dp("world_offset", [0.0, 0.0]).value)
        self.table_z = float(dp("table_z", float("nan")).value)  # nan: from the first coarse estimate
        self.table_z_expected = float(dp("table_z_expected", -0.206).value)
        self.table_z_tolerance = float(dp("table_z_tolerance", 0.03).value)
        self.workspace_lo = np.array(dp("workspace_lo", [-0.9, -0.9, -1.0]).value, float)  # fingertips, base_link
        self.workspace_hi = np.array(dp("workspace_hi", [0.9, 0.9, 0.6]).value, float)
        self.sizes = {c: tuple(dp(f"{c}_size", list(s)).value) for c, s in
                      (("green", (0.05, 0.05)), ("black", (0.04, 0.08)), ("red", (0.035, 0.03)))}  # (w, h)
        self.velocity_scaling = float(dp("velocity_scaling", 0.1).value)
        self.acceleration_scaling = float(dp("acceleration_scaling", 0.1).value)
        self.settle = float(dp("settle", 1.0).value)
        self.estimate_frames = int(dp("estimate_frames", 5).value)
        self.estimate_timeout = float(dp("estimate_timeout", 6.0).value)
        self.avoid_collisions = bool(dp("avoid_collisions", True).value)
        self.base_frame = dp("base_frame", "base_link").value
        self.wrist_frame = dp("wrist_frame", "wrist_3_link").value
        self.group = dp("group", "ur_manipulator").value

        cb = ReentrantCallbackGroup()
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 10, callback_group=cb)
        self.create_subscription(WrenchStamped, dp("wrench_topic", "/force_torque_sensor_broadcaster/wrench").value,
                                 self.wrench_cb, 50, callback_group=cb)
        self.create_subscription(MarkerArray, "/tower/blocks", self.blocks_cb, 10, callback_group=cb)
        self.create_subscription(Bool, "/io_and_status_controller/robot_program_running", self.program_cb, 10,
                                 callback_group=cb)
        self.traj_pub = self.create_publisher(JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory",
                                              1)
        self.target_pub = self.create_publisher(PoseStamped, "/tower/target", 1)
        self.ik_cli = self.create_client(GetPositionIK, "/compute_ik", callback_group=cb)
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path", callback_group=cb)
        self.gripper_cli = self.create_client(Trigger, dp("gripper_service", "/toggle_gripper").value,
                                              callback_group=cb)
        self.switch_cli = self.create_client(SwitchController, "/controller_manager/switch_controller",
                                             callback_group=cb)
        self.exec_ac = ActionClient(self, FollowJointTrajectory,
                                    "/scaled_joint_trajectory_controller/follow_joint_trajectory", callback_group=cb)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lock = threading.Lock()
        self.joints = None
        self.joints_t = 0.0
        self.start_joints = None
        self.program_running = None
        self.forces = deque(maxlen=500)  # (monotonic time, force xyz in the wrench frame)
        self.wrench_frame = None
        self.block_msgs = deque(maxlen=200)
        self.active_goal = None
        self.streaming = False
        self.dry_q = None
        self.dry_tool = None

        self.cfg = SimConfig.from_run(self.policy_path.parent)
        self.policy = load_policy(self.policy_path)
        self.ScriptedStack = scripted_stack_class(self.repo)
        self.gripper = GripperModel(self.gripper_threshold)  # /toggle_gripper only toggles: start open
        self.frame = None

    # ------------------------------------------------------------ callbacks

    def joint_cb(self, msg: JointState):
        if not set(UR_JOINTS) <= set(msg.name):
            return
        self.joints, self.joints_t = msg, time.monotonic()
        if self.start_joints is None:
            self.start_joints = msg

    def wrench_cb(self, msg: WrenchStamped):
        f = msg.wrench.force
        with self.lock:
            self.wrench_frame = msg.header.frame_id
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

    def program_cb(self, msg: Bool):
        was, self.program_running = self.program_running, msg.data
        if was and not msg.data:
            self.get_logger().error("robot program stopped (e-stop, protective stop or pendant)")

    # ------------------------------------------------------------ setup

    def wait_ready(self):
        end = time.monotonic() + 10.0
        while self.start_joints is None:
            if time.monotonic() > end:
                raise PolicyAbort("no /joint_states")
            time.sleep(0.05)
        for cli, name in ((self.ik_cli, "/compute_ik"), (self.plan_cli, "/plan_kinematic_path")):
            if not cli.wait_for_service(timeout_sec=10.0):
                raise PolicyAbort(f"no {name}: is MoveIt up?")
        if self.dry_run:
            return
        time.sleep(0.5)
        if self._recent_forces() is None:
            raise PolicyAbort("no wrench on the wrench topic")
        if self.program_running is False:
            raise PolicyAbort("the robot program is not running: press play on the pendant")
        if self.switch_cli.wait_for_service(timeout_sec=5.0):
            req = SwitchController.Request()
            req.activate_controllers = ["scaled_joint_trajectory_controller"]
            req.deactivate_controllers = ["freedrive_mode_controller", "forward_position_controller",
                                          "forward_velocity_controller"]
            req.strictness = SwitchController.Request.BEST_EFFORT
            req.activate_asap = True
            res = wait(self.switch_cli.call_async(req), 5.0)
            if res is None or not res.ok:
                self.get_logger().warn("Could not switch to scaled_joint_trajectory_controller")

    # ------------------------------------------------------------ the run

    def run(self):
        log = self.get_logger()
        start = joint_vector(self.start_joints)
        stages = STAGES[:max(1, min(self.stages, len(STAGES)))]
        for k, (held, target) in enumerate(stages):
            if k > 0:
                self.go_joints(start, "back to the start pose for the coarse estimate")
            est = self.estimate(BLOCKS, "coarse")
            for c in (held, target):
                if c not in est:
                    raise PolicyAbort(f"the {c} block is not in view: check /tower/block_points")
            if self.frame is None:
                self.frame = self._make_frame(est)
            close = {}
            if self.closeup and not self.dry_run:
                for c in (held, target):
                    look = tool_for_camera_above((est[c].x, est[c].y), est[c].top + self.closeup_height,
                                                 self._tool().yaw, self.gripper_length, self.camera_offset)
                    self.go_tool(look, f"camera above the {c} block")
                    got = self.estimate((c,), "close-up")
                    if c not in got:
                        raise PolicyAbort(f"no close-up of the {c} block")
                    close[c] = got[c]
                    jump = math.hypot(close[c].x - est[c].x, close[c].y - est[c].y)
                    log.info(f"close-up moved the {c} block {jump * 1000:.1f} mm")
                    if jump > 0.05:
                        raise PolicyAbort(f"the {c} close-up is {jump * 1000:.0f} mm from the coarse estimate")
            elif self.closeup:
                close = dict(est)  # dry run: no move, the coarse estimate stands in
            if not self.run_stage(k, held, target, est, close):
                raise PolicyAbort(f"stage {k} ({held} on {target}) did not finish in {self.step_limit} steps")
            log.info(f"stage {k}: {held} released on {target}")
        self.go_joints(start, "back to the start pose")
        for c, e in self.estimate(BLOCKS, "final").items():
            log.info(f"  {c}: ({e.x:.3f}, {e.y:.3f}) top {e.top:.3f}")

    def _make_frame(self, est) -> PolicyFrame:
        table_z = self.table_z
        if math.isnan(table_z):
            table_z = float(np.median([e.bottom for e in est.values()]))
            if abs(table_z - self.table_z_expected) > self.table_z_tolerance:
                raise PolicyAbort(f"table measured at z {table_z:.3f}, expected {self.table_z_expected:.3f}: "
                                  "set table_z if it really moved")
        offset = self.world_offset
        if self.auto_offset:
            center = PolicyFrame(table_z, self.world_yaw).to_world(
                np.mean([[e.x, e.y, e.bottom] for e in est.values()], 0))
            offset = tuple((np.add(self.cfg.xy_lo, self.cfg.xy_hi) / 2 - center[:2]).tolist())
        frame = PolicyFrame(table_z, self.world_yaw, offset, self.finger_axis_yaw)
        self.get_logger().info(f"table at z {table_z:.3f} in {self.base_frame}, "
                               f"policy frame offset ({offset[0]:.3f}, {offset[1]:.3f})")
        for c, e in est.items():
            p = frame.to_world([e.x, e.y, e.bottom])
            if not (self.cfg.xy_lo[0] - 0.05 <= p[0] <= self.cfg.xy_hi[0] + 0.05
                    and self.cfg.xy_lo[1] - 0.05 <= p[1] <= self.cfg.xy_hi[1] + 0.05):
                self.get_logger().warn(f"the {c} block is at ({p[0]:.2f}, {p[1]:.2f}) in the policy frame, "
                                       f"trained on x {self.cfg.xy_lo[0]}..{self.cfg.xy_hi[0]}, "
                                       f"y {self.cfg.xy_lo[1]}..{self.cfg.xy_hi[1]}: move it or set world_offset")
        return frame

    def run_stage(self, stage: int, held: str, target: str, est: dict, close: dict) -> bool:
        log, cfg, fr = self.get_logger(), self.cfg, self.frame
        heights = {c: s[1] for c, s in self.sizes.items()}
        below = stack_below(target, heights)

        def hold(e):
            p, yaw = fr.block(e, 0.0, heights[held])
            return p + [0, 0, self.grasp_raise], yaw

        def top(e):
            p, yaw = fr.block(e, below, heights[target])
            return p + [0, 0, heights[target] / 2 + self.grasp_raise], yaw

        held_e, target_top = hold(est[held]), top(est[target])
        held_close = hold(close[held]) if held in close else None
        target_close = top(close[target]) if target in close else None
        if target_close is not None:
            camera_top = fr.to_world([0, 0, close[target].top])[2]
            ruler_top = target_close[0][2] - self.grasp_raise
            if abs(camera_top - ruler_top) > 0.015:
                log.warn(f"the camera puts the {target} top at {camera_top:.3f}, the ruler heights at "
                         f"{ruler_top:.3f} above the table: is the tower where it should be?")

        tallest = max(fr.to_world([0, 0, e.top])[2] for e in est.values())
        start_tcp = sim_start(held_e[0], heights[held], tallest, cfg)
        _, yaw_now = fr.tcp_from_tool(self._tool())
        self.go_tool(fr.tool_from_tcp(start_tcp, yaw_now), "sim start over the held block")

        self._confirm(f"stage {stage}: {held} on {target}, policy loop at "
                      f"{1 / (cfg.control_dt * self.time_scale):.0f} Hz")
        tare = self._tare()
        tcp, yaw = fr.tcp_from_tool(self._tool())
        s = StageState(cfg, stage, held_e, target_top, self.sizes[held], self.sizes[target][0], tcp, yaw,
                       held_close, target_close)
        s.force = self._policy_force(tare, yaw)
        s.wrist_look(tcp)
        script = self.ScriptedStack()
        base = script(s)
        q_cmd = self._q()
        close_z = None
        period = cfg.control_dt * self.time_scale
        max_step, max_yaw_step = self.max_speed * period, self.max_yaw_rate * period
        self.streaming = not self.dry_run
        next_t = time.monotonic()
        try:
            while s.t < self.step_limit:
                obs = residual_obs(s.obs(tcp, yaw, self.gripper.opening), base, script.phase)
                residual = np.asarray(self.policy(obs), float)
                a = total_action(base, residual, RESIDUAL_SCALE)
                pos, tyaw = step_target(s.target_pos, s.target_yaw, a, cfg, max_step, max_yaw_step)
                tool = fr.tool_from_tcp(pos, tyaw)
                xyz = np.clip(tool.xyz(), self.workspace_lo, self.workspace_hi)
                if not np.allclose(xyz, tool.xyz()):
                    tool = ToolPose(*xyz, tool.yaw)
                    pos = fr.tcp_from_tool(tool)[0]
                s.target_pos, s.target_yaw = pos, tyaw
                q_cmd = self._stream(tool, q_cmd, period)
                log.info(f"t {s.t:3d} phase {script.phase} a {np.round(a, 2)} res {np.round(residual, 2)} "
                         f"tcp {np.round(tcp, 3)} f {np.round(s.force, 1)}"
                         + (f" wrist {int(s.wrist_held)}{int(s.wrist_target)}" if s.t % 10 == 0 else ""))

                want = self.gripper.wanted(a[4])
                if want is not None:
                    width = self.sizes[held][0] if want and closing_on_block(tcp, s.est_grasp) else 0.0
                    self.toggle(want, width, f"{'close on' if want else 'open from'} the {held} block")
                    close_z = tcp[2] if want else None
                    next_t = time.monotonic()

                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
                if self._arrive(tool, tare):
                    next_t = time.monotonic()
                s.t += 1
                tcp, yaw = fr.tcp_from_tool(self._tool())
                self._check(tool, tare)
                s.force = self._policy_force(tare, yaw)
                if (self.dry_run and self.gripper.width and tcp[2] < s.est_place[2]
                        and np.linalg.norm(tcp[:2] - s.est_place[:2]) < 0.01):
                    s.force = s.force + [0.0, 0.0, 5.0]  # the held block meets the estimated top
                if self.gripper.width and close_z is not None and tcp[2] > close_z + 0.01:
                    s.lifted = True
                s.wrist_look(tcp)
                s.prev_action = a
                base = script(s)
                if stage_done(script, s):
                    log.info(f"stage {stage} done in {s.t} steps (sim: ~85)")
                    return True
            return False
        finally:
            self.hold()

    # ------------------------------------------------------------ motion

    def _stream(self, tool: ToolPose, q_cmd, period: float):
        self._publish_target(tool)
        q = self._ik(tool, q_cmd, timeout=0.5 * period)
        if q is None:
            raise PolicyAbort(f"no IK for ({tool.x:.3f}, {tool.y:.3f}, {tool.z:.3f}) yaw {tool.yaw:.2f}")
        jump = float(np.max(np.abs(q - q_cmd)))
        if jump > self.max_joint_step:
            raise PolicyAbort(f"IK jumps {jump:.2f} rad in one period")
        if self.dry_run:
            self.dry_q, self.dry_tool = q, tool
            return q
        if self.program_running is False:
            raise PolicyAbort("the robot program stopped")
        pt = JointTrajectoryPoint(positions=[float(v) for v in q], time_from_start=duration(self.lookahead * period))
        self.traj_pub.publish(JointTrajectory(joint_names=list(UR_JOINTS), points=[pt]))
        return q

    def _arrive(self, tool: ToolPose, tare) -> bool:
        """Waits for the fingertips to reach the command; True if it had to wait. Stops waiting on contact
        (the held block on the tower never reaches a command below it) and leaves the guards to _check."""
        if self.dry_run:
            return False
        end, waited = time.monotonic() + self.arrive_timeout, False
        while np.linalg.norm(self._tool().xyz() - tool.xyz()) > self.arrive_tol and time.monotonic() < end:
            forces = self._recent_forces()
            if self.program_running is False or forces is None or np.linalg.norm(forces.mean(0) - tare) > 3.0:
                return waited
            time.sleep(0.01)
            waited = True
        return waited

    def _check(self, tool: ToolPose, tare):
        """After each period: the arm follows, the wrench is alive and under guard_force."""
        if self.dry_run:
            return
        if time.monotonic() - self.joints_t > 0.5:
            raise PolicyAbort("/joint_states stopped")
        if self.program_running is False:
            raise PolicyAbort("the robot program stopped (e-stop or protective stop)")
        err = float(np.linalg.norm(self._tool().xyz() - tool.xyz()))
        if err > self.tracking_tol:
            raise PolicyAbort(f"the fingertips are {err * 1000:.0f} mm from the command: the arm is not following")
        forces = self._recent_forces()
        if forces is None:
            raise PolicyAbort("the wrench topic stopped")
        f = float(np.linalg.norm(forces.mean(0) - tare))
        if f > self.guard_force:
            raise PolicyAbort(f"{f:.1f} N on the wrist, over guard_force {self.guard_force:.0f} N")

    def hold(self):
        """Stop streaming: one point at the measured joints, the controller holds there."""
        if not self.streaming:
            return
        self.streaming = False
        if self.joints is not None:
            pt = JointTrajectoryPoint(positions=[float(v) for v in joint_vector(self.joints)],
                                      time_from_start=duration(0.1))
            self.traj_pub.publish(JointTrajectory(joint_names=list(UR_JOINTS), points=[pt]))

    def go_tool(self, pose: ToolPose, label: str):
        q = self._ik(pose, self._q(), timeout=1.0)
        if q is None:
            raise PolicyAbort(f"no IK for {label}")
        self.go_joints(q, f"{label}: fingertips ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})", pose)

    def go_joints(self, q, label: str, pose: ToolPose = None):
        if pose is not None:
            self._publish_target(pose)
        if self.dry_run:
            self.get_logger().info(f"[dry run] {label}")
            self.dry_q, self.dry_tool = np.asarray(q, float), pose
            return
        traj = self._plan(q)
        d = traj.points[-1].time_from_start if traj.points else DurationMsg()
        self._confirm(f"{label} (planned, {d.sec + d.nanosec * 1e-9:.1f} s)")
        if not self.exec_ac.wait_for_server(timeout_sec=5.0):
            raise PolicyAbort("no trajectory controller action server")
        gh = wait(self.exec_ac.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj)), 5.0)
        if gh is None or not gh.accepted:
            raise PolicyAbort("the trajectory controller rejected the goal")
        self.active_goal = gh
        result = gh.get_result_async()
        forces = self._recent_forces()
        if forces is None:
            self.stop_motion()
            raise PolicyAbort("the wrench topic stopped")
        tare = forces.mean(0)
        while not result.done():
            forces = self._recent_forces()
            if forces is None or np.linalg.norm(forces.mean(0) - tare) > self.guard_force:
                self.stop_motion()
                raise PolicyAbort("wrench lost or over guard_force during a planned move")
            time.sleep(0.005)
        self.active_goal = None
        res = result.result()
        if res.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise PolicyAbort(f"trajectory failed ({res.result.error_code}): {res.result.error_string}")

    def stop_motion(self):
        gh, self.active_goal = self.active_goal, None
        if gh is not None:
            wait(gh.cancel_goal_async(), 2.0)
        self.hold()

    def toggle(self, close: bool, width: float, label: str):
        """The loop pauses for the toggle (the sim's fingers take 5 periods, the Hand-E about a second)."""
        self.gripper.set(close, width)
        if self.dry_run:
            self.get_logger().info(f"[dry run] toggle gripper: {label}")
            return
        self.get_logger().info(f"toggle gripper: {label}")
        if not self.gripper_cli.wait_for_service(timeout_sec=2.0):
            raise PolicyAbort("no /toggle_gripper service")
        res = wait(self.gripper_cli.call_async(Trigger.Request()), 5.0)
        if res is None:
            raise PolicyAbort("/toggle_gripper did not answer")
        if not res.success:
            self.get_logger().warn(f"/toggle_gripper: {res.message}")
        time.sleep(self.gripper_wait)

    # ------------------------------------------------------------ sensing

    def estimate(self, colors, kind: str) -> dict:
        t0 = self.get_clock().now().nanoseconds + int(self.settle * 1e9)
        end = time.monotonic() + self.settle + self.estimate_timeout
        samples, used = {c: [] for c in colors}, set()
        while time.monotonic() < end and any(len(v) < self.estimate_frames for v in samples.values()):
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
        out = {c: combine_samples(v) for c, v in samples.items() if len(v) >= max(1, self.estimate_frames // 2)}
        for c, e in out.items():
            self.get_logger().info(f"{kind} {c}: ({e.x:.3f}, {e.y:.3f}) top {e.top:.3f} bottom {e.bottom:.3f} "
                                   f"yaw {math.degrees(e.yaw):.1f} deg")
        return out

    def _q(self) -> np.ndarray:
        if self.dry_run and self.dry_q is not None:
            return self.dry_q
        return joint_vector(self.joints)

    def _tool(self) -> ToolPose:
        """Fingertip pose from TF; in a dry run the last command, as if the arm tracked it exactly."""
        if self.dry_run and self.dry_tool is not None:
            return self.dry_tool
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.wrist_frame, Time())
        except TransformException as ex:
            raise PolicyAbort(f"no TF {self.base_frame} -> {self.wrist_frame}: {ex}")
        t, q = tf.transform.translation, tf.transform.rotation
        return tool_from_wrist((t.x, t.y, t.z), (q.x, q.y, q.z, q.w), self.gripper_length)

    def _recent_forces(self, n=None):
        n = n or self.force_samples
        with self.lock:
            if not self.forces or time.monotonic() - self.forces[-1][0] > 0.2:
                return None
            return np.array([f for _, f in list(self.forces)[-n:]])

    def _tare(self):
        """The zero at stage start, as the sim tares its sensor in _start_stage."""
        time.sleep(0.3)
        forces = self._recent_forces(n=50)
        if forces is None:
            if self.dry_run:
                return np.zeros(3)
            raise PolicyAbort("no wrench to zero")
        return forces.mean(0)

    def _policy_force(self, tare, yaw_world: float) -> np.ndarray:
        """The wrist force in the sim's sensor axes; read in a dry run too, for the force_sign check."""
        forces = self._recent_forces()
        if forces is None:
            if self.dry_run:
                return np.zeros(3)
            raise PolicyAbort("the wrench topic stopped")
        f = forces.mean(0) - tare
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.wrench_frame, Time())
        except TransformException as ex:
            raise PolicyAbort(f"no TF {self.base_frame} -> {self.wrench_frame}: {ex}")
        r = tf.transform.rotation
        f_base = quat_to_matrix((r.x, r.y, r.z, r.w)) @ f
        return sensor_force(f_base, yaw_world, self.frame, self.force_sign)

    # ------------------------------------------------------------ MoveIt

    def _ik(self, pose: ToolPose, seed, timeout: float):
        xyz, quat = wrist_from_tool(pose, self.gripper_length)
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.group
        ik.ik_link_name = self.wrist_frame
        ik.avoid_collisions = self.avoid_collisions
        ik.timeout = duration(min(timeout, 1.0))
        ik.robot_state.joint_state = joint_state(seed)
        ik.pose_stamped.header.frame_id = self.base_frame
        p, o = ik.pose_stamped.pose.position, ik.pose_stamped.pose.orientation
        p.x, p.y, p.z = (float(v) for v in xyz)
        o.x, o.y, o.z, o.w = (float(v) for v in quat)
        res = wait(self.ik_cli.call_async(req), timeout + 0.05)
        if res is None or res.error_code.val != res.error_code.SUCCESS:
            return None
        return joint_vector(res.solution.joint_state)

    def _plan(self, q) -> JointTrajectory:
        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = self.group
        mpr.allowed_planning_time = 5.0
        mpr.num_planning_attempts = 3
        mpr.max_velocity_scaling_factor = self.velocity_scaling
        mpr.max_acceleration_scaling_factor = self.acceleration_scaling
        goal = Constraints()
        for name, pos in zip(UR_JOINTS, q):
            goal.joint_constraints.append(JointConstraint(joint_name=name, position=float(pos), tolerance_above=0.01,
                                                          tolerance_below=0.01, weight=1.0))
        mpr.goal_constraints.append(goal)
        res = wait(self.plan_cli.call_async(req), 15.0)
        if res is None or res.motion_plan_response.error_code.val != 1:
            raise PolicyAbort("MoveIt found no plan")
        return res.motion_plan_response.trajectory.joint_trajectory

    def _publish_target(self, pose: ToolPose):
        xyz, q = wrist_from_tool(pose, self.gripper_length)
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in xyz)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q
        self.target_pub.publish(msg)

    def _confirm(self, desc: str):
        if not self.step or self.dry_run:
            self.get_logger().info(desc)
            return
        answer = input(f"\n[step] {desc}\n       Enter to run, q to stop: ")
        if answer.strip().lower().startswith("q"):
            raise PolicyAbort("stopped by the operator")


def main(args=None):
    # Ctrl-C raises KeyboardInterrupt here without shutting the context down, so the hold point still goes out.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = TowerPolicy()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    log = node.get_logger()
    try:
        node.wait_ready()
        log.info(f"Tower policy {node.policy_path} on the real arm, {'dry run' if node.dry_run else 'LIVE'}, "
                 f"time_scale {node.time_scale}, max_speed {node.max_speed} m/s. The gripper must be open now.")
        node.run()
        log.info("Tower done")
    except PolicyAbort as e:
        log.error(f"Stopped: {e}")
    except KeyboardInterrupt:
        log.warn("Interrupted")
    finally:
        try:
            node.stop_motion()
        except Exception as e:  # the context may already be down after Ctrl-C: the e-stop is the real stop
            log.error(f"Could not stop the arm: {e}")
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
