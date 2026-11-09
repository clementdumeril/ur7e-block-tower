"""Geometry of the block tower on the real UR7e, free of ROS so the unit tests run on any machine.

Frames: everything is in base_link. The tool pose is the fingertip center with the gripper pointing
down, turned by `yaw` about the vertical; MoveIt gets wrist_3_link, GRIPPER_LENGTH above the fingertips.
"""

import math
from dataclasses import dataclass

import numpy as np

BLOCKS = ("green", "black", "red")  # bottom to top, as in the sim
STAGES = (("black", "green"), ("red", "black"))  # (held, target)


def wrap_quarter(angle: float) -> float:
    """Angle modulo a quarter turn, in [-45, 45) deg: a square block looks the same every 90 deg."""
    return (angle + math.pi / 4) % (math.pi / 2) - math.pi / 4


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class BlockEstimate:
    x: float  # top-face center
    y: float
    top: float  # z of the top face
    yaw: float  # top-face edge direction, modulo 90 deg
    height: float  # top minus the table around the block
    n_points: int = 0

    @property
    def bottom(self) -> float:
        return self.top - self.height


@dataclass
class ToolPose:
    x: float  # fingertip center
    y: float
    z: float
    yaw: float

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


# ---------------------------------------------------------------- orientation

def quat_gripper_down(yaw: float) -> tuple:
    """(x, y, z, w) of Rz(yaw) * (0, 1, 0, 0): the gripper-down quaternion turned about the vertical."""
    return (-math.sin(yaw / 2), math.cos(yaw / 2), 0.0, 0.0)


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_yaw(yaw: float) -> tuple:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def gripper_down_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-c, -s, 0.0], [-s, c, 0.0], [0.0, 0.0, -1.0]])


def wrist_from_tool(pose: ToolPose, gripper_length: float):
    """wrist_3_link position and quaternion for a fingertip pose (fingertips on the wrist axis)."""
    return np.array([pose.x, pose.y, pose.z + gripper_length]), quat_gripper_down(pose.yaw)


def tool_from_wrist(position, quat, gripper_length: float) -> ToolPose:
    """Fingertip pose from a wrist_3_link pose read on TF; the yaw assumes the gripper points down."""
    r = quat_to_matrix(quat)
    tip = np.asarray(position, float) + r @ [0.0, 0.0, gripper_length]
    return ToolPose(*tip, math.atan2(-r[1, 0], -r[0, 0]))


def tool_for_camera_above(xy, camera_z: float, yaw: float, gripper_length: float, camera_offset) -> ToolPose:
    """Tool pose that puts the wrist camera straight above `xy` at height `camera_z`.

    The RealSense sits camera_offset from wrist_3_link ((-0.025, 0.13, 0) on the lab arm, optical axis along the
    wrist z), so with the fingertips above a block the camera looks 13 cm to the side of it. For the
    close-up the camera, not the gripper, goes above the block.
    """
    off = gripper_down_matrix(yaw) @ np.asarray(camera_offset, float)
    wrist = np.array([xy[0] - off[0], xy[1] - off[1], camera_z - off[2]])
    return ToolPose(wrist[0], wrist[1], wrist[2] - gripper_length, yaw)


def grip_yaw(block_yaw: float, finger_axis_yaw: float = 0.0) -> float:
    """Tool yaw that lines the fingers up with the block faces, within 45 deg of zero."""
    return wrap_quarter(block_yaw - finger_axis_yaw)


def place_yaw(grasp_tool_yaw: float, held_yaw: float, target_yaw: float) -> float:
    """Turn the tool by the smallest quarter-turn-equivalent angle that lines the held block up with the target."""
    return grasp_tool_yaw + wrap_quarter(target_yaw - held_yaw)


def linear_waypoints(start: ToolPose, end: ToolPose, step: float, yaw_step: float = math.radians(3)):
    """Straight line from start to end (start excluded, end included), at most `step` apart."""
    dyaw = wrap_pi(end.yaw - start.yaw)
    dist = float(np.linalg.norm(end.xyz() - start.xyz()))
    n = max(1, math.ceil(dist / step - 1e-9), math.ceil(abs(dyaw) / yaw_step - 1e-9))
    a, b = start.xyz(), end.xyz()
    return [ToolPose(*(a + (b - a) * k / n), start.yaw + dyaw * k / n) for k in range(1, n + 1)]


def waypoint_times(start: ToolPose, poses, speed: float, yaw_rate: float = 0.3, min_dt: float = 0.05):
    """Time from start of each waypoint at `speed` m/s and at most `yaw_rate` rad/s."""
    t, prev, out = 0.0, start, []
    for p in poses:
        dt = max(np.linalg.norm(p.xyz() - prev.xyz()) / speed, abs(wrap_pi(p.yaw - prev.yaw)) / yaw_rate, min_dt)
        t += dt
        out.append(t)
        prev = p
    return out


def contact_detected(forces, baseline, threshold: float) -> bool:
    """Mean of the latest wrench samples against the baseline taken at rest, so one noisy sample does not
    stop the arm. The norm keeps it independent of the frame the wrench is published in."""
    forces = np.asarray(forces, float).reshape(-1, 3)
    if not len(forces):
        return False
    return float(np.linalg.norm(forces.mean(0) - np.asarray(baseline, float))) > threshold


# ---------------------------------------------------------------- point cloud

def transform_points(points: np.ndarray, translation, quat) -> np.ndarray:
    return points @ quat_to_matrix(quat).T + np.asarray(translation, float)


def unpack_rgb(packed: np.ndarray) -> np.ndarray:
    """RealSense packed 'rgb' float field to (N, 3) uint8 R, G, B."""
    u = np.ascontiguousarray(packed, dtype=np.float32).view(np.uint32)
    return np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], axis=1).astype(np.uint8)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 RGB to HSV in OpenCV's 8-bit convention (H 0..179, S and V 0..255),
    without needing cv2."""
    c = rgb.astype(np.float32)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    v = c.max(1)
    diff = v - c.min(1)
    s = np.where(v > 0, diff * 255.0 / np.maximum(v, 1e-6), 0.0)
    d = np.maximum(diff, 1e-6)
    h = np.where(v == r, 60.0 * (g - b) / d, np.where(v == g, 120.0 + 60.0 * (b - r) / d, 240.0 + 60.0 * (r - g) / d))
    h = np.where(diff == 0, 0.0, h) % 360.0
    return np.stack([np.round(h / 2) % 180, np.round(s), v], axis=1).astype(np.uint8)


@dataclass
class HsvThresholds:
    # z band above the table where block points can be (m)
    band_min: float = 0.008
    band_max: float = 0.20
    green_h_min: int = 40
    green_h_max: int = 85
    green_s_min: int = 80
    green_v_min: int = 40
    # red wraps around hue 0
    red_h_max: int = 10
    red_h_min: int = 170
    red_s_min: int = 120
    red_v_min: int = 50
    # black by low V
    black_v_max: int = 80


def color_masks(hsv: np.ndarray, z: np.ndarray, table_z: float, t: HsvThresholds) -> dict:
    band = (z > table_z + t.band_min) & (z < table_z + t.band_max)
    h, s, v = (hsv[:, k].astype(np.int32) for k in range(3))
    green = band & (h >= t.green_h_min) & (h <= t.green_h_max) & (s >= t.green_s_min) & (v >= t.green_v_min)
    red = band & ((h <= t.red_h_max) | (h >= t.red_h_min)) & (s >= t.red_s_min) & (v >= t.red_v_min)
    black = band & (v <= t.black_v_max) & ~green & ~red
    return {"green": green, "black": black, "red": red}


def table_height(z: np.ndarray, lo: float, hi: float, bin_size: float = 0.005):
    """Table z as the most populated height between lo and hi: from above, the table fills most of the view."""
    z = z[(z > lo) & (z < hi)]
    if len(z) < 100:
        return None
    counts, edges = np.histogram(z, bins=max(1, int((hi - lo) / bin_size)), range=(lo, hi))
    k = int(np.argmax(counts))
    near = z[(z > edges[k] - bin_size) & (z < edges[k + 1] + bin_size)]
    return float(np.median(near))


def _min_area_rect_np(xy: np.ndarray):
    """Smallest bounding rectangle by search over directions; the fallback when cv2 is missing."""
    mean = xy.mean(0)
    pts = xy - mean

    def best(angles):
        dirs = np.stack([np.cos(angles), np.sin(angles)])
        u, v = pts @ dirs, pts @ np.stack([-dirs[1], dirs[0]])
        area = np.ptp(u, 0) * np.ptp(v, 0)
        return angles[int(np.argmin(area))]

    th = best(np.radians(np.arange(0.0, 90.0, 1.0)))
    th = best(th + np.radians(np.arange(-1.0, 1.0001, 0.02)))
    c, s = math.cos(th), math.sin(th)
    u, v = pts @ [c, s], pts @ [-s, c]
    uc, vc = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    center = mean + uc * np.array([c, s]) + vc * np.array([-s, c])
    return (float(center[0]), float(center[1])), (float(np.ptp(u)), float(np.ptp(v))), th


def top_face_rect(xy: np.ndarray, use_cv2: bool = True):
    """Center, size and yaw (modulo 90 deg) of the top-face points, from cv2.minAreaRect."""
    xy = np.asarray(xy, float)
    if use_cv2:
        try:
            import cv2
        except ImportError:
            use_cv2 = False
    if use_cv2:
        (cx, cy), (w, l), ang = cv2.minAreaRect(xy.astype(np.float32))
        return (float(cx), float(cy)), (float(w), float(l)), wrap_quarter(math.radians(ang))
    center, size, th = _min_area_rect_np(xy)
    return center, size, wrap_quarter(th)


def estimate_block(points: np.ndarray, mask: np.ndarray, any_block: np.ndarray, table_z: float,
                   table_annulus=(0.06, 0.15), table_tol: float = 0.03, top_band: float = 0.008,
                   cluster_radius: float = 0.06, min_points: int = 30, top_offset: float = 0.0):
    """One block from its color mask: the top is a high percentile of the
    block points, the center comes from the top face only (the sides the camera sees would pull it toward
    the camera), and the height is the top minus the median of the table points in an annulus around the block."""
    pts = points[mask]
    if len(pts) < min_points:
        return None
    # one block per color: drop stray points of that color away from the main blob
    med = np.median(pts[:, :2], 0)
    pts = pts[np.linalg.norm(pts[:, :2] - med, axis=1) < cluster_radius]
    if len(pts) < min_points:
        return None
    top = float(np.percentile(pts[:, 2], 95))
    face = pts[pts[:, 2] > top - top_band]
    if len(face) < 5:
        return None
    (cx, cy), _, yaw = top_face_rect(face[:, :2])
    d = np.linalg.norm(points[:, :2] - [cx, cy], axis=1)
    sel = (d >= table_annulus[0]) & (d <= table_annulus[1]) & ~any_block & (np.abs(points[:, 2] - table_z) < table_tol)
    local_table = float(np.median(points[sel, 2])) if sel.sum() >= min_points else table_z
    top += top_offset
    return BlockEstimate(cx, cy, top, yaw, top - local_table, int(len(pts)))


def combine_samples(samples) -> BlockEstimate:
    """Median over several frames; the yaw through the mean of 4*yaw, so 44 and -44 deg average to 45."""
    a = np.array([[s.x, s.y, s.top, s.height] for s in samples])
    x, y, top, height = np.median(a, 0)
    yaws = np.array([s.yaw for s in samples])
    yaw = wrap_quarter(math.atan2(np.sin(4 * yaws).mean(), np.cos(4 * yaws).mean()) / 4)
    return BlockEstimate(float(x), float(y), float(top), yaw, float(height), int(np.median([s.n_points for s in samples])))


def marker_from_estimate(e: BlockEstimate):
    """RViz CUBE marker center and z size: a box from the table up to the measured top."""
    return (e.x, e.y, e.top - e.height / 2), e.height


def estimate_from_marker(position, orientation, scale_z: float, n_points: int = 0) -> BlockEstimate:
    x, y, z = position
    qx, qy, qz, qw = orientation
    yaw = wrap_quarter(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return BlockEstimate(x, y, z + scale_z / 2, yaw, scale_z, n_points)
