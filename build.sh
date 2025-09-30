#!/usr/bin/env bash
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-jazzy}"
WS="${WS:-$PWD/ws}"

# Source base ROS if present
if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
  source "/opt/ros/$ROS_DISTRO/setup.bash"
fi

# Install deps (rosdep will skip if nothing to do)
if command -v rosdep >/dev/null; then
  sudo rosdep init 2>/dev/null || true
  rosdep update
  rosdep install --from-paths "$WS/src" --ignore-src -r -y --rosdistro "$ROS_DISTRO"
fi

pushd "$WS" >/dev/null
colcon build --symlink-install --event-handlers console_cohesion+
popd >/dev/null

echo
echo "Build complete. To use this overlay:"
echo "    source $WS/install/setup.bash"
