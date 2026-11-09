"""The phase logic of pensig/tower_scripted.py (ScriptedStack) for the real arm.

It is written against a small robot interface so the same code drives the arm (tower_node.TowerRobot),
the dry run and the unit tests. Per stage (black on green, then red on black):

  coarse          estimate from the start pose; a stage already stacked (a run stopped by the e-stop and
                  started again) is skipped
  closeup_held    camera above the held block, second estimate (the sim's wrist close-up: 52 % -> 93 %)
  above_held      fingertips pre_grasp above the block top
  grasp_descent   straight down to finger_above_bottom over the block bottom, at least table_clearance
  close           /toggle_gripper
  lift            straight up until the block clears everything
  closeup_target  camera above the target, second estimate of its top
  above_target    over the target at the same height
  approach        down to slow_zone above the estimated place height, stops on contact
  touchdown       wrench zeroed, down at slow_speed until contact, at most search_below past the estimate
  open            /toggle_gripper
  retract         straight up
"""

import math
from dataclasses import dataclass, field

from tower_ros.geometry import BLOCKS, STAGES, BlockEstimate, ToolPose, grip_yaw, place_yaw, tool_for_camera_above


class TowerAbort(RuntimeError):
    """Stop where the arm is and leave it to the operator."""


@dataclass
class TowerParams:
    # wrist_3_link (the flange) to fingertips: 0.16 measured on the lab Hand-E. Too short puts the
    # fingers into the table, so check it on the arm.
    gripper_length: float = 0.16
    finger_above_bottom: float = 0.015  # sim cfg.finger_above_bottom
    table_clearance: float = 0.02  # fingertips above the table, at least
    clearance: float = 0.05  # the held block's bottom above every other block when it moves across
    pre_grasp: float = 0.05  # fingertips above the block top before the straight descent
    second_estimate: bool = True
    closeup_height: float = 0.30  # camera above the block top for the second estimate
    camera_offset: tuple = (-0.025, 0.13, 0.0)  # wrist_3_link -> camera_depth_optical_frame, as the static TF
    finger_axis_yaw: float = 0.0  # direction the fingers close along at tool yaw 0, in base_link
    move_speed: float = 0.03  # m/s for straight moves
    approach_speed: float = 0.02
    slow_zone: float = 0.012  # last stretch above the estimate done at slow_speed (sim: 12 mm at 1 cm/s)
    slow_speed: float = 0.01
    search_below: float = 0.004  # how far past the estimated top the touchdown may go
    contact_force: float = 3.0  # N above the zeroed wrench
    approach_force: float = 6.0
    retract: float = 0.08
    max_estimate_jump: float = 0.05  # a close-up farther than this from the coarse estimate is another object
    block_heights: dict = field(default_factory=lambda: {"green": 0.05, "black": 0.05, "red": 0.04})
    height_tolerance: float = 0.008
    # Already stacked: the held block within stacked_xy of the target and standing at least stacked_rise
    # taller above the table than it is (it sits on something).
    stacked_xy: float = 0.035
    stacked_rise: float = 0.02


class Robot:
    """What the script needs from the arm. Every call returns once the motion is over."""

    def go_start(self, label: str) -> None:
        raise NotImplementedError

    def estimate(self, colors, close: bool) -> dict:
        """{color: BlockEstimate} averaged over a few frames; colors not seen are left out."""
        raise NotImplementedError

    def move(self, pose: ToolPose, label: str, speed=None, stop_force=None) -> bool:
        """speed None: MoveIt-planned joint move. Otherwise a straight line at `speed` m/s. With stop_force,
        stop (not abort) once the wrench is that far above the last zero. Returns whether it stopped on contact."""
        raise NotImplementedError

    def zero_wrench(self) -> None:
        raise NotImplementedError

    def gripper(self, close: bool, label: str) -> None:
        raise NotImplementedError


class TowerScript:
    def __init__(self, robot: Robot, params: TowerParams = None, log=print):
        self.r = robot
        self.p = params or TowerParams()
        self.log = log
        self.phases = []

    def phase(self, name: str, detail: str = ""):
        self.phases.append(name)
        self.log(f"[{name}] {detail}".rstrip())

    def run(self, stages=STAGES) -> dict:
        for held, target in stages:
            self.stack(held, target)
        self.phase("done")
        self.r.go_start("back to the start pose")
        final = self.r.estimate(BLOCKS, close=False)
        for color, e in final.items():
            self.log(f"  {color}: ({e.x:.3f}, {e.y:.3f}) top {e.top:.3f}")
        return final

    def stack(self, held: str, target: str):
        p, r = self.p, self.r
        self.phase("coarse", f"{held} on {target}")
        r.go_start("start pose for the coarse estimate")
        est = r.estimate(BLOCKS, close=False)
        h, t = self._need(est, held), self._need(est, target)
        if self._stacked(held, h, t):
            self.phase("skip", f"the {held} block is already on the {target} block")
            return
        self._check_height(held, h)
        others_top = max(e.top for c, e in est.items() if c != held)

        yaw_g = grip_yaw(h.yaw, p.finger_axis_yaw)
        if p.second_estimate:
            self.phase("closeup_held")
            r.move(self._camera_above(h, yaw_g, h.top + p.pre_grasp), f"camera above the {held} block")
            h = self._refine(r.estimate((held,), close=True), held, h)
            self._check_height(held, h)
            yaw_g = grip_yaw(h.yaw, p.finger_axis_yaw)

        self.phase("above_held")
        r.move(ToolPose(h.x, h.y, h.top + p.pre_grasp, yaw_g), f"fingertips above the {held} block",
               speed=p.move_speed if p.second_estimate else None)

        # h.bottom is the table next to the block, so the grasp does not depend on the measured top,
        # which the RealSense gets wrong on the black block.
        grasp_z = max(h.bottom + p.finger_above_bottom, h.bottom + p.table_clearance)
        grasp_height = grasp_z - h.bottom
        self.phase("grasp_descent", f"fingertips {grasp_height * 1000:.0f} mm above the block bottom")
        r.move(ToolPose(h.x, h.y, grasp_z, yaw_g), "down to the grasp", speed=p.approach_speed)
        self.phase("close")
        r.gripper(True, f"close on the {held} block")

        carry_z = max(others_top, t.top) + grasp_height + p.clearance
        self.phase("lift")
        r.move(ToolPose(h.x, h.y, carry_z, yaw_g), "lift", speed=p.move_speed)
        hover_z = carry_z

        if p.second_estimate:
            self.phase("closeup_target")
            look = self._camera_above(t, yaw_g, carry_z)
            r.move(look, f"camera above the {target} block", speed=p.move_speed)
            t = self._refine(r.estimate((target,), close=True), target, t)
            hover_z = look.z

        yaw_p = place_yaw(yaw_g, h.yaw, t.yaw)
        place_z = t.top + grasp_height  # fingertips when the held block sits on the target
        self.phase("above_target", f"turn {math.degrees(yaw_p - yaw_g):+.1f} deg")
        r.move(ToolPose(t.x, t.y, hover_z, yaw_p), f"over the {target} block", speed=p.move_speed)

        self.phase("approach")
        r.zero_wrench()
        touched = r.move(ToolPose(t.x, t.y, place_z + p.slow_zone, yaw_p), "down to the slow zone",
                         speed=p.approach_speed, stop_force=p.approach_force)
        if touched:
            self.log("  contact before the slow zone: the target top is higher than estimated")
        else:
            self.phase("touchdown")
            r.zero_wrench()
            touched = r.move(ToolPose(t.x, t.y, place_z - p.search_below, yaw_p), "touchdown",
                             speed=p.slow_speed, stop_force=p.contact_force)
            if not touched:
                self.log(f"  no contact down to {p.search_below * 1000:.0f} mm below the estimate, releasing there")

        self.phase("open")
        r.gripper(False, f"release the {held} block")
        self.phase("retract")
        r.move(ToolPose(t.x, t.y, place_z + p.retract, yaw_p), "retract", speed=p.move_speed)

    # ------------------------------------------------------------ helpers

    def _camera_above(self, e: BlockEstimate, yaw: float, min_z: float) -> ToolPose:
        p = self.p
        pose = tool_for_camera_above((e.x, e.y), e.top + p.closeup_height, yaw, p.gripper_length, p.camera_offset)
        pose.z = max(pose.z, min_z)
        return pose

    def _need(self, est: dict, color: str) -> BlockEstimate:
        if color not in est:
            raise TowerAbort(f"the {color} block is not in view: check the HSV thresholds on /tower/block_points")
        return est[color]

    def _refine(self, est: dict, color: str, old: BlockEstimate) -> BlockEstimate:
        new = self._need(est, color)
        dxy = math.hypot(new.x - old.x, new.y - old.y)
        dz = new.top - old.top
        self.log(f"  close-up moved the {color} block {dxy * 1000:.1f} mm in xy, {dz * 1000:+.1f} mm in z")
        if dxy > self.p.max_estimate_jump or abs(dz) > self.p.max_estimate_jump:
            raise TowerAbort(f"the close-up {color} estimate is {dxy * 1000:.0f} mm / {dz * 1000:+.0f} mm "
                             "from the first one: probably another object")
        return new

    def _stacked(self, color: str, h: BlockEstimate, t: BlockEstimate) -> bool:
        # The height is the top minus the table next to the block, so a block on another one reads taller.
        # Its top is a poorer test: the camera reads dark tops low, and the covered target's center is off.
        want = self.p.block_heights.get(color)
        if want is None:
            return False
        return math.hypot(h.x - t.x, h.y - t.y) < self.p.stacked_xy and h.height > want + self.p.stacked_rise

    def _check_height(self, color: str, e: BlockEstimate):
        want = self.p.block_heights.get(color)
        if want is not None and abs(e.height - want) > self.p.height_tolerance:
            self.log(f"  warning: the {color} block measures {e.height * 1000:.0f} mm tall, expected "
                     f"{want * 1000:.0f} mm")
