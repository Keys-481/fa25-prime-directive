import rclpy
from std_msgs.msg import String
from potato_mission.mission_node import MissionNode

def test_initial_state_is_flight():
    rclpy.init()
    node = MissionNode()
    # Give the timer a tick to publish once
    rclpy.spin_once(node, timeout_sec=0.1)
    # Subscribe briefly and fetch one message
    got = []

    def cb(msg):
        got.append(msg.data)

    sub = node.create_subscription(String, "/mission/state", cb, 10)
    rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()
    assert "FLIGHT" in got
