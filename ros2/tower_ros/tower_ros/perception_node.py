"""Finds the green, black and red blocks in the wrist RealSense cloud by color and height above the table.

Per block and frame: top-face center and yaw on /tower/<color>/top (PoseStamped, base_link), the height
(top minus the table around the block) on /tower/<color>/height, and all blocks together on /tower/blocks
(MarkerArray, one CUBE per block from the table to the top) which the tower node reads.

Black is found by low V. The RealSense puts the black block's top too low (black absorbs the
projector's IR); measure it and set black_top_offset. The tower node takes the block bottom from the
table around it, so the grasp does not depend on it, but the place on black does.
"""

import math

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from tower_ros.geometry import (BLOCKS, HsvThresholds, color_masks, estimate_block, marker_from_estimate,
                                quat_yaw, rgb_to_hsv, table_height, transform_points, unpack_rgb)

MARKER_RGB = {"green": (0.2, 0.8, 0.3), "black": (0.15, 0.15, 0.15), "red": (0.9, 0.1, 0.1)}


class TowerPerception(Node):
    def __init__(self):
        super().__init__("tower_perception")
        dp = self.declare_parameter
        self.base_frame = dp("base_frame", "base_link").value
        self.cloud_topic = dp("cloud_topic", "/camera/camera/depth/color/points").value
        self.max_rate = float(dp("max_rate", 5.0).value)  # Hz, the cloud comes at 30 Hz
        self.stride = int(dp("stride", 2).value)  # keep every stride-th point
        # where to look in base_link (max_y 0.79 keeps the wall out)
        self.x_lim = (float(dp("min_x", -1.0).value), float(dp("max_x", 1.0).value))
        self.y_lim = (float(dp("min_y", -1.0).value), float(dp("max_y", 0.79).value))
        # table z: auto = most populated height between table_min_z and table_max_z, else table_z
        self.table_auto = bool(dp("table_auto", True).value)
        self.table_z = float(dp("table_z", -0.19).value)
        self.table_lim = (float(dp("table_min_z", -0.40).value), float(dp("table_max_z", 0.10).value))
        self.table_annulus = (float(dp("table_annulus_min", 0.06).value),
                              float(dp("table_annulus_max", 0.15).value))
        self.min_points = int(dp("min_points", 30).value)
        self.top_band = float(dp("top_band", 0.008).value)
        # added to the measured top of each block (m); the black top reads low
        self.top_offset = {c: float(dp(f"{c}_top_offset", 0.0).value) for c in BLOCKS}
        # drop points near the gripper axis below the wrist (black fingers look like the black block)
        self.wrist_frame = dp("wrist_frame", "wrist_3_link").value
        self.finger_radius = float(dp("finger_radius", 0.04).value)
        self.gripper_length = float(dp("gripper_length", 0.16).value)

        t = HsvThresholds()
        self.hsv = HsvThresholds(**{k: type(v)(dp(k, v).value) for k, v in vars(t).items()})

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PointCloud2, self.cloud_topic, self.cloud_cb, 1)
        self.blocks_pub = self.create_publisher(MarkerArray, "/tower/blocks", 1)
        self.points_pub = self.create_publisher(PointCloud2, "/tower/block_points", 1)
        self.top_pub = {c: self.create_publisher(PoseStamped, f"/tower/{c}/top", 1) for c in BLOCKS}
        self.height_pub = {c: self.create_publisher(Float32, f"/tower/{c}/height", 1) for c in BLOCKS}
        self.last_t = 0.0
        self.get_logger().info(f"Looking for {', '.join(BLOCKS)} blocks on {self.cloud_topic}")

    def _lookup(self, target, source, stamp):
        tf = self.tf_buffer.lookup_transform(target, source, stamp)
        tr, q = tf.transform.translation, tf.transform.rotation
        return (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)

    def cloud_cb(self, msg: PointCloud2):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_t < 1.0 / self.max_rate:
            return
        self.last_t = now
        if "rgb" not in [f.name for f in msg.fields]:
            self.get_logger().warn("The cloud has no rgb field: launch the RealSense with pointcloud.enable",
                                   throttle_duration_sec=5.0)
            return
        # The cloud's own stamp, so a moving arm does not shift the blocks; the latest TF
        # if that one is not in the buffer yet. The tower node only reads frames taken at rest anyway.
        try:
            try:
                stamp = Time.from_msg(msg.header.stamp)
                cam = self._lookup(self.base_frame, msg.header.frame_id, stamp)
                wrist = self._lookup(self.base_frame, self.wrist_frame, stamp)
            except TransformException:
                cam = self._lookup(self.base_frame, msg.header.frame_id, Time())
                wrist = self._lookup(self.base_frame, self.wrist_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(f"No TF for the cloud: {ex}", throttle_duration_sec=2.0)
            return

        raw = pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=False).reshape(-1)
        raw = raw[:: max(1, self.stride)]
        xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
        valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
        points = transform_points(xyz[valid], *cam)
        hsv = rgb_to_hsv(unpack_rgb(raw["rgb"][valid]))

        roi = ((points[:, 0] >= self.x_lim[0]) & (points[:, 0] <= self.x_lim[1])
               & (points[:, 1] >= self.y_lim[0]) & (points[:, 1] <= self.y_lim[1]))
        if self.finger_radius > 0:
            w = np.asarray(wrist[0])
            near_axis = np.linalg.norm(points[:, :2] - w[:2], axis=1) < self.finger_radius
            roi &= ~(near_axis & (points[:, 2] > w[2] - self.gripper_length - 0.01))
        points, hsv = points[roi], hsv[roi]

        table_z = self.table_z
        if self.table_auto:
            found = table_height(points[:, 2], *self.table_lim)
            if found is None:
                self.get_logger().warn("No table in view", throttle_duration_sec=2.0)
                return
            table_z = found

        masks = color_masks(hsv, points[:, 2], table_z, self.hsv)
        any_block = masks["green"] | masks["black"] | masks["red"]
        header = msg.header
        header.frame_id = self.base_frame
        if any_block.any():
            self.points_pub.publish(pc2.create_cloud_xyz32(header, points[any_block].astype(np.float32).tolist()))

        markers = MarkerArray()
        for k, color in enumerate(BLOCKS):
            e = estimate_block(points, masks[color], any_block, table_z, table_annulus=self.table_annulus, top_band=self.top_band,
                               min_points=self.min_points, top_offset=self.top_offset[color])
            m = Marker(header=header, ns=color, id=k, type=Marker.CUBE)
            if e is None:
                m.action = Marker.DELETE
                markers.markers.append(m)
                continue
            q = quat_yaw(e.yaw)
            self.height_pub[color].publish(Float32(data=float(e.height)))
            top = PoseStamped(header=header)
            top.pose.position.x, top.pose.position.y, top.pose.position.z = e.x, e.y, e.top
            (top.pose.orientation.x, top.pose.orientation.y,
             top.pose.orientation.z, top.pose.orientation.w) = q
            self.top_pub[color].publish(top)

            center, size_z = marker_from_estimate(e)
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = center
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = q
            m.scale.x = m.scale.y = 0.04
            m.scale.z = max(float(size_z), 1e-3)
            m.color.r, m.color.g, m.color.b = MARKER_RGB[color]
            m.color.a = 0.6
            m.text = str(e.n_points)  # ignored by RViz for a CUBE, read by the tower node
            markers.markers.append(m)
            self.get_logger().debug(f"{color}: ({e.x:.3f}, {e.y:.3f}) top {e.top:.3f} "
                                    f"h {e.height * 1000:.0f} mm yaw {math.degrees(e.yaw):.1f}")
        self.blocks_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = TowerPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
