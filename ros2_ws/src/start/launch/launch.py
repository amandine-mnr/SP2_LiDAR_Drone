from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, RegisterEventHandler, Shutdown
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit

import os
from pathlib import Path

current = Path(__file__).resolve()

for parent in current.parents:
    if parent.name == "SP2_LiDAR":
        MAIN_DIRECTORY = parent
        break
else:
    raise RuntimeError("Could not find SP2_LiDAR")

bridge_file_path = os.path.join(MAIN_DIRECTORY, "ros2_ws/bridge.yaml")

def generate_launch_description():

    gz_resource_path = os.path.join(MAIN_DIRECTORY, "crazyflie-simulation/simulator_files/gazebo/")

    # Supervisor node
    supervisor = Node(
        package="start",
        executable="supervisor_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
        ]
    )

    # Localization node
    localizer = Node(
        package="lidar",
        executable="localization_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            os.path.join(MAIN_DIRECTORY, "ros2_ws/src/config/param.yaml")
        ],
    )

    # Shutdown when localizer exits
    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=localizer,
            on_exit=[Shutdown()]
        )
    )

    return LaunchDescription([

        # Gazebo environment variable
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=gz_resource_path
        ),

        # Gazebo Sim
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "worlds/crazyflie_world_main.sdf"],
            output="screen"
        ),

        # ROS2 - Gazebo bridges
        ExecuteProcess(
            cmd=[
                "ros2", "run", "ros_gz_bridge", "parameter_bridge",
                "--ros-args",
                "-p", f"config_file:={bridge_file_path}"
            ],
            output="screen"
        ),

        supervisor,

        localizer,

        shutdown_handler,

        # Trajectory controller
        Node(
            package="crazyflie",
            executable="traj_controller_node",
            output="screen",
            parameters=[
                {"use_sim_time": True},
                os.path.join(MAIN_DIRECTORY, "ros2_ws/src/config/param.yaml")
            ],
        ),

        

        # RViz
        ExecuteProcess(
            cmd=[
                "rviz2",
                "-d",
                os.path.join(MAIN_DIRECTORY, "rviz_config.rviz")
            ],
            output="screen"
        ),
    ])