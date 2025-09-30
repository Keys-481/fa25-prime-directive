from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="potato_mission",
            executable="mission_node",
            name="mission",
            parameters=[{"initial_state": "FLIGHT", "heartbeat_period_s": 0.25}],
            output="screen",
        )
    ])
