import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare launch arguments
    node_name_arg = DeclareLaunchArgument("node_name", default_value="sentra_ros")

    # Set path to the config file
    config_file_path = os.path.join(
        get_package_share_directory("sentra_ros"), "config", "params.yml"
    )

    return LaunchDescription(
        [
            node_name_arg,
            Node(
                package="sentra_ros",
                executable="sentra_node",
                name=LaunchConfiguration("node_name"),
                output="screen",
                parameters=[
                    config_file_path,
                ],
            ),
        ]
    )
