#!/usr/bin/env python3
import enum
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rcl_interfaces.msg import SetParametersResult

class RoverState(str, enum.Enum):
    FLIGHT = "FLIGHT"
    DEPLOYED = "EXPLORE"
    EXPLORE = "NAVIGATE"

class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")
        self.declare_parameter("initial_state", RoverState.FLIGHT.value)
        self.declare_parameter("heartbeat_period_s", 0.5)

        init = self.get_parameter("initial_state").get_parameter_value().string_value
        self._state = RoverState(init) if init in RoverState.__members__.values() else RoverState.FLIGHT

        self.state_pub = self.create_publisher(String, "/mission/state", 10)

        period = float(self.get_parameter("heartbeat_period_s").value)
        self.timer = self.create_timer(period, self._tick)

        self.add_on_set_parameters_callback(self._on_params)

        self.get_logger().info(f"Mission node up. Initial state = {self._state}")

    def _on_params(self, params):
        for p in params:
            if p.name == "initial_state":
                try:
                    self._state = RoverState(p.value)
                    self.get_logger().info(f"Initial state overridden to {self._state}")
                except Exception:
                    return SetParametersResult(successful=False, reason="invalid state")
            if p.name == "heartbeat_period_s":
                try:
                    self.timer.timer_period_ns = int(float(p.value) * 1e9)
                except Exception:
                    return SetParametersResult(successful=False, reason="bad period")
        return SetParametersResult(successful=True)

    def _tick(self):
        msg = String()
        msg.data = self._state.value
        self.state_pub.publish(msg)

def main():
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
