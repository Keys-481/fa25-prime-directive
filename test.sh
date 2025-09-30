#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
WS="${WS:-$PWD/ws}"

# Prefer overlay if built, else base
if [ -f "$WS/install/setup.bash" ]; then
  source "$WS/install/setup.bash"
elif [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
  source "/opt/ros/$ROS_DISTRO/setup.bash"
fi

pushd "$WS" >/dev/null
colcon test --event-handlers console_cohesion+ || { echo "Tests failed"; exit 1; }
colcon test-result --verbose
popd >/dev/null
echo "Tests complete."
