"""Camera, wrist TF, block perception and MoveIt for the tower.

The tower node itself reads Enter from the terminal in step mode, so it runs in its own terminal:
  ros2 run tower_ros tower --ros-args --params-file $(ros2 pkg prefix tower_ros)/share/tower_ros/config/tower.yaml
"""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# wrist_3_link -> camera_depth_optical_frame, identity rotation, as the lab workspace's planning tf node.
# Keep it equal to camera_offset in config/tower.yaml.
CAMERA_OFFSET = ("-0.025", "0.13", "0.0")


def generate_launch_description():
    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true")
    config = os.path.join(get_package_share_directory("tower_ros"), "config", "tower.yaml")

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")),
        launch_arguments={
            "pointcloud.enable": "true",
            "rgb_camera.color_profile": "1920x1080x30",
        }.items(),
    )

    # The RealSense publishes its own camera_link -> camera_depth_frame -> camera_depth_optical_frame tree. A
    # static wrist_3_link -> camera_depth_optical_frame then races its static parent, so the depth frame can
    # end up hanging off camera_link, away from the arm. The lab workspace's planning tf node re-publishes the
    # wrist link at 20 Hz and wins that race: use it when it is there.
    try:
        get_package_share_directory("planning")
        camera_tf = Node(package="planning", executable="tf", name="wrist_camera_tf", output="screen")
    except PackageNotFoundError:
        camera_tf = Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="wrist_camera_tf",
            arguments=["--x", CAMERA_OFFSET[0], "--y", CAMERA_OFFSET[1], "--z", CAMERA_OFFSET[2],
                       "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                       "--frame-id", "wrist_3_link", "--child-frame-id", "camera_depth_optical_frame"],
            output="screen",
        )

    perception_node = Node(
        package="tower_ros",
        executable="perception",
        name="tower_perception",
        parameters=[config],
        output="screen",
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ur_moveit_config"), "launch", "ur_moveit.launch.py")),
        launch_arguments={"ur_type": ur_type, "launch_rviz": launch_rviz}.items(),
    )

    actions = [realsense_launch, camera_tf, perception_node, moveit_launch]

    # Table, wall and gripper collision boxes, when the lab workspace's planning package is there.
    try:
        planning_share = get_package_share_directory("planning")
        actions.append(Node(
            package="planning",
            executable="planning_scene_obstacles",
            name="planning_scene_obstacles",
            output="screen",
            parameters=[{"obstacles_config": os.path.join(planning_share, "config", "obstacles.yaml")}],
        ))
    except PackageNotFoundError:
        print("[tower_bringup] 'planning' package not found: no collision boxes in MoveIt")

    # Global shutdown on any process exit
    actions.append(RegisterEventHandler(
        OnProcessExit(on_exit=[EmitEvent(event=Shutdown(reason="A launched process exited"))])))
    return LaunchDescription(actions)
