from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit


def generate_launch_description():

    config_file = LaunchConfiguration("config_file")

    # Localization node
    localizer = Node(
        package="lidar",
        executable="localization_node",
        output="screen",
        parameters=[{"use_sim_time": True}, config_file],
    )

    # Shutdown when localizer exits
    shutdown_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=localizer,
            on_exit=[Shutdown()]
        )
    )

    actions = [
        DeclareLaunchArgument(
            "config_file",
            default_value="/home/ameunier/ros2_ws/src/config/param.yaml",
        ),

       
        # Gazebo Sim
        ExecuteProcess(
            cmd=["gz", "sim", "-s", "-r", "/home/ameunier/crazyflie-simulation/simulator_files/gazebo/worlds/crazyflie_world.sdf"],
            output="screen"
        ),


        # Supervisor
        Node(
            package="start",
            executable="supervisor_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
        ),

        # Trajectory controller
        Node(
            package="crazyflie",
            executable="traj_controller_node",
            output="screen",
            parameters=[{"use_sim_time": True}, config_file],
        ),


        localizer,

        shutdown_handler,
        
    ]

    return LaunchDescription(actions)
