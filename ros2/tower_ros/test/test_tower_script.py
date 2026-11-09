"""The tower phase logic against a fake arm: the camera sees exact poses only from the close-up, the
held block follows the fingers, and a downward move stops where the held block meets the target."""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tower_ros.geometry import BlockEstimate, ToolPose, gripper_down_matrix, wrap_quarter  # noqa: E402
from tower_ros.script import Robot, TowerAbort, TowerParams, TowerScript  # noqa: E402

STAGE_PHASES = ["coarse", "closeup_held", "above_held", "grasp_descent", "close", "lift", "closeup_target",
                "above_target", "approach", "touchdown", "open", "retract"]


@dataclass
class Block:
    x: float
    y: float
    bottom: float
    height: float
    yaw: float

    @property
    def top(self):
        return self.bottom + self.height


class FakeArm(Robot):
    def __init__(self, p: TowerParams, coarse_err=(0.008, -0.006, -0.004, 0.06), top_err=None, hide=()):
        self.p = p
        self.blocks = {
            "green": Block(0.45, -0.10, 0.0, 0.050, math.radians(12)),
            "black": Block(0.55, 0.12, 0.0, 0.050, math.radians(-30)),
            "red": Block(0.35, 0.08, 0.0, 0.040, math.radians(40)),
        }
        self.coarse_err = coarse_err
        self.top_err = top_err or {}  # color -> error the camera always makes on the top (the black block)
        self.hide = set(hide)
        self.start = ToolPose(0.40, 0.0, 0.40, 0.0)
        self.tool = self.start
        self.calls = []
        self.closed = False
        self.held = None
        self.grasp_h = None
        self.zeroed = False
        self.closeups = []

    def camera(self):
        wrist = self.tool.xyz() + [0, 0, self.p.gripper_length]
        return wrist + gripper_down_matrix(self.tool.yaw) @ np.asarray(self.p.camera_offset)

    def go_start(self, label):
        self.calls.append(("start",))
        self.tool = self.start

    def estimate(self, colors, close):
        self.calls.append(("estimate", tuple(colors), close))
        cam = self.camera()
        out = {}
        for c in colors:
            b = self.blocks[c]
            if c in self.hide or c == self.held:
                continue
            top = b.top + self.top_err.get(c, 0.0)
            if close and math.hypot(cam[0] - b.x, cam[1] - b.y) < 0.02:
                self.closeups.append(c)
                out[c] = BlockEstimate(b.x, b.y, top, wrap_quarter(b.yaw), top - b.bottom)
            else:
                ex, ey, ez, eyaw = self.coarse_err
                out[c] = BlockEstimate(b.x + ex, b.y + ey, top + ez, wrap_quarter(b.yaw + eyaw), top + ez - b.bottom)
        return out

    def move(self, pose, label, speed=None, stop_force=None):
        self.calls.append(("move", label, speed, stop_force))
        assert pose.z > 0.0, f"{label}: fingertips into the table"
        touched = False
        if self.held is not None:
            hb = self.blocks[self.held]
            for c, b in self.blocks.items():
                if c == self.held or math.hypot(pose.x - b.x, pose.y - b.y) > 0.02:
                    continue
                contact_z = b.top + self.grasp_h
                if pose.z < contact_z - 1e-9:
                    assert stop_force, f"{label}: the held block would crash into the {c} block"
                    assert self.zeroed
                    pose = ToolPose(pose.x, pose.y, contact_z, pose.yaw)
                    touched = True
            hb.x, hb.y, hb.bottom = pose.x, pose.y, pose.z - self.grasp_h
            hb.yaw = hb.yaw + (pose.yaw - self.tool.yaw)
        self.tool = pose
        return touched

    def zero_wrench(self):
        self.calls.append(("zero",))
        self.zeroed = True

    def gripper(self, close, label):
        self.calls.append(("gripper", close))
        assert close != self.closed
        self.closed = close
        if close:
            for c, b in self.blocks.items():
                if math.hypot(self.tool.x - b.x, self.tool.y - b.y) < 0.004 and b.bottom < self.tool.z < b.top:
                    assert abs(wrap_quarter(b.yaw - self.tool.yaw)) < math.radians(2), "fingers not on the faces"
                    self.held, self.grasp_h = c, self.tool.z - b.bottom
                    return
            raise AssertionError(f"closed on nothing at {self.tool}")
        hb = self.blocks[self.held]
        below = [b.top for c, b in self.blocks.items()
                 if c != self.held and math.hypot(hb.x - b.x, hb.y - b.y) < 0.02] or [0.0]
        self.drop = hb.bottom - max(below)
        hb.bottom = max(below)
        self.held = None
        self.zeroed = False


def test_one_stage_phase_sequence():
    p = TowerParams()
    arm = FakeArm(p)
    script = TowerScript(arm, p, log=lambda *_: None)
    script.stack("black", "green")
    assert script.phases == STAGE_PHASES
    assert arm.closeups == ["black", "green"]
    grip = [c for c in arm.calls if c[0] == "gripper"]
    assert grip == [("gripper", True), ("gripper", False)]
    # the touchdown is the only move at slow speed, with the contact threshold, after a zero
    k = [i for i, c in enumerate(arm.calls) if c[0] == "move" and c[1] == "touchdown"][0]
    assert arm.calls[k][2:] == (p.slow_speed, p.contact_force)
    assert arm.calls[k - 1] == ("zero",)


def test_grasp_height_and_alignment():
    p = TowerParams()
    arm = FakeArm(p)
    TowerScript(arm, p, log=lambda *_: None).stack("black", "green")
    # table_clearance (20 mm) wins over finger_above_bottom (15 mm) for a block on the table
    assert arm.grasp_h == pytest.approx(p.table_clearance)
    black, green = arm.blocks["black"], arm.blocks["green"]
    assert math.hypot(black.x - green.x, black.y - green.y) < 1e-6
    assert black.bottom == pytest.approx(green.top)
    assert abs(wrap_quarter(black.yaw - green.yaw)) < 1e-6
    assert arm.drop == pytest.approx(0.0)  # released on contact, not from above


def test_full_tower():
    p = TowerParams()
    arm = FakeArm(p)
    script = TowerScript(arm, p, log=lambda *_: None)
    script.run()
    assert script.phases == STAGE_PHASES * 2 + ["done"]
    g, k, r = (arm.blocks[c] for c in ("green", "black", "red"))
    assert math.hypot(k.x - g.x, k.y - g.y) < 1e-6 and math.hypot(r.x - k.x, r.y - k.y) < 1e-6
    assert k.bottom == pytest.approx(g.top) and r.bottom == pytest.approx(k.top)
    assert r.top == pytest.approx(0.14)


def test_without_second_estimate_the_coarse_error_stays():
    p = TowerParams(second_estimate=False)
    arm = FakeArm(p, coarse_err=(0.0, 0.0, 0.0, 0.0))
    script = TowerScript(arm, p, log=lambda *_: None)
    script.stack("black", "green")
    assert script.phases == [s for s in STAGE_PHASES if not s.startswith("closeup")]
    assert arm.closeups == []
    # the first move goes to the block through MoveIt
    first = [c for c in arm.calls if c[0] == "move"][0]
    assert first[2] is None


def test_target_top_read_low_contacts_during_approach():
    # The camera reads the black top 2 cm low: the place on black meets it before the slow zone.
    p = TowerParams()
    arm = FakeArm(p, top_err={"black": -0.02})
    script = TowerScript(arm, p, log=lambda *_: None)
    script.stack("black", "green")
    script.phases.clear()
    script.stack("red", "black")
    assert "touchdown" not in script.phases
    r, k = arm.blocks["red"], arm.blocks["black"]
    assert r.bottom == pytest.approx(k.top)
    approach = [c for c in arm.calls if c[0] == "move" and c[1] == "down to the slow zone"][-1]
    assert approach[3] == p.approach_force


def test_missing_block_aborts():
    p = TowerParams()
    with pytest.raises(TowerAbort, match="red"):
        TowerScript(FakeArm(p, hide={"red"}), p, log=lambda *_: None).stack("red", "black")


def test_close_up_far_from_coarse_aborts():
    p = TowerParams()
    arm = FakeArm(p, coarse_err=(0.0, 0.0, 0.0, 0.0))
    script = TowerScript(arm, p, log=lambda *_: None)
    real_estimate = arm.estimate

    def jumpy(colors, close):
        out = real_estimate(colors, close)
        if close:
            for e in out.values():
                e.x += 0.08
        return out

    arm.estimate = jumpy
    with pytest.raises(TowerAbort, match="another object"):
        script.stack("black", "green")
    assert not arm.closed


def test_height_warning():
    p = TowerParams()
    arm = FakeArm(p, coarse_err=(0.0, 0.0, 0.015, 0.0))
    logs = []
    TowerScript(arm, p, log=logs.append).stack("black", "green")
    assert any("measures 65 mm" in m for m in logs)


class MeasuresFromTable(FakeArm):
    """The real camera measures a block's height from the table next to it, so a stacked block reads taller."""

    def estimate(self, colors, close):
        out = super().estimate(colors, close)
        for c, e in out.items():
            e.height = e.top
        return out


def test_stacked_stage_is_skipped():
    # A run stopped after the first placement and started again: black on green is not redone.
    p = TowerParams()
    arm = MeasuresFromTable(p, coarse_err=(0.0, 0.0, 0.0, 0.0))
    TowerScript(arm, p, log=lambda *_: None).stack("black", "green")
    arm.tool = arm.start
    script = TowerScript(arm, p, log=lambda *_: None)
    script.run()
    assert script.phases[:2] == ["coarse", "skip"]
    assert script.phases.count("close") == 1  # only the red block is picked
    r, k = arm.blocks["red"], arm.blocks["black"]
    assert r.bottom == pytest.approx(k.top)


def test_block_on_the_table_is_not_stacked():
    p = TowerParams()
    script = TowerScript(FakeArm(p), p, log=lambda *_: None)
    green = BlockEstimate(0.45, -0.10, 0.05, 0.0, 0.05)
    assert not script._stacked("black", BlockEstimate(0.45, -0.10, 0.05, 0.0, 0.05), green)
    assert not script._stacked("black", BlockEstimate(0.55, 0.12, 0.10, 0.0, 0.10), green)
    assert script._stacked("black", BlockEstimate(0.452, -0.101, 0.10, 0.0, 0.10), green)
