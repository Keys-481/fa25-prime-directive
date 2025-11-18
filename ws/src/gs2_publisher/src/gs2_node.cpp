#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <src/CYdLidar.h>
#include <core/common/ydlidar_def.h>

using namespace std::chrono_literals;

namespace {

void set_str(CYdLidar & lidar, int prop, const std::string & value, const rclcpp::Logger & logger) {
  if (!lidar.setlidaropt(prop, value.c_str(), value.size())) {
    RCLCPP_WARN(logger, "Failed to set string prop %d", prop);
  }
}

void set_int(CYdLidar & lidar, int prop, int value, const rclcpp::Logger & logger) {
  if (!lidar.setlidaropt(prop, &value, sizeof(int))) {
    RCLCPP_WARN(logger, "Failed to set int prop %d", prop);
  }
}

void set_bool(CYdLidar & lidar, int prop, bool value, const rclcpp::Logger & logger) {
  if (!lidar.setlidaropt(prop, &value, sizeof(bool))) {
    RCLCPP_WARN(logger, "Failed to set bool prop %d", prop);
  }
}

void set_float(CYdLidar & lidar, int prop, float value, const rclcpp::Logger & logger) {
  if (!lidar.setlidaropt(prop, &value, sizeof(float))) {
    RCLCPP_WARN(logger, "Failed to set float prop %d", prop);
  }
}

}  // namespace

class GS2Node : public rclcpp::Node {
public:
  GS2Node()
  : rclcpp::Node("gs2_publisher")
  {
    // Parameters
    port_     = declare_parameter<std::string>("port", "/dev/ydlidar");
    baud_     = declare_parameter<int>("baudrate", 921600);
    frame_    = declare_parameter<std::string>("frame_id", "laser_frame");
    amin_deg_ = declare_parameter<double>("angle_min_deg", -50.0);
    amax_deg_ = declare_parameter<double>("angle_max_deg", 50.0);
    rmin_     = declare_parameter<double>("range_min", 0.025);
    rmax_     = declare_parameter<double>("range_max", 1.0);

    pub_ = create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);

    const auto logger = get_logger();

    // string props
    set_str(laser_, LidarPropSerialPort, port_, logger);
    const std::string ignore_array;  // empty
    set_str(laser_, LidarPropIgnoreArray, ignore_array, logger);

    // int props
    set_int(laser_, LidarPropSerialBaudrate, baud_, logger);

    int lidar_type = TYPE_GS;
    set_int(laser_, LidarPropLidarType, lidar_type, logger);

    int device_type = YDLIDAR_TYPE_SERIAL;  // serial mode
    set_int(laser_, LidarPropDeviceType, device_type, logger);

    int sample_rate = 4;  // 4 kHz (per sample)
    set_int(laser_, LidarPropSampleRate, sample_rate, logger);

    int abnormal_check_count = 4;
    set_int(laser_, LidarPropAbnormalCheckCount, abnormal_check_count, logger);

    int intensity_bits = 8;
    set_int(laser_, LidarPropIntenstiyBit, intensity_bits, logger);

    // bool props
    set_bool(laser_, LidarPropFixedResolution, false, logger);
    set_bool(laser_, LidarPropReversion, false, logger);
    set_bool(laser_, LidarPropInverted, false, logger);
    set_bool(laser_, LidarPropAutoReconnect, true, logger);

    bool is_single_channel = false;
    set_bool(laser_, LidarPropSingleChannel, is_single_channel, logger);

    set_bool(laser_, LidarPropIntenstiy, true, logger);
    set_bool(laser_, LidarPropSupportMotorDtrCtrl, true, logger);
    set_bool(laser_, LidarPropSupportHeartBeat, false, logger);

    // float props
    set_float(laser_, LidarPropMaxAngle, 180.0f, logger);
    set_float(laser_, LidarPropMinAngle, -180.0f, logger);
    set_float(laser_, LidarPropMaxRange, static_cast<float>(rmax_), logger);
    set_float(laser_, LidarPropMinRange, static_cast<float>(rmin_), logger);

    float freq = 8.0f;
    set_float(laser_, LidarPropScanFrequency, freq, logger);

    laser_.setEnableDebug(true);

    RCLCPP_INFO(
      logger, "Opening GS2 (TYPE_GS) on %s @ %d", port_.c_str(),
      baud_);

    if (!laser_.initialize()) {
      throw std::runtime_error(
        std::string("initialize() failed: ") + laser_.DescribeError());
    }

    // small settle
    rclcpp::sleep_for(200ms);

    if (!laser_.turnOn()) {
      throw std::runtime_error(
        std::string("turnOn() failed: ") + laser_.DescribeError());
    }

    // light polling loop
    timer_ = create_wall_timer(5ms, [this]() { tick(); });
  }

  ~GS2Node() override {
    try {
      laser_.turnOff();
      laser_.disconnecting();
    } catch (...) {
      // best-effort shutdown
    }
  }

private:
  void tick() {
    LaserScan scan;
    if (!laser_.doProcessSimple(scan)) {
      // No data this cycle.
      return;
    }

    // publish only requested window
    const std::size_t N = scan.points.size();
    if (N > 2) {
      std::reverse(scan.points.begin() + (N / 2), scan.points.end());
      std::reverse(scan.points.begin(), scan.points.begin() + (N / 2));
    }

    const double amin = amin_deg_ * M_PI / 180.0;
    const double amax = amax_deg_ * M_PI / 180.0;

    std::vector<float> ranges;
    ranges.reserve(scan.points.size());

    for (const auto & p : scan.points) {
      if (p.angle < amin || p.angle > amax) {
        continue;
      }
      float r = static_cast<float>(p.range);
      if (r <= 0.0f) {
        r = std::numeric_limits<float>::infinity();
      }
      ranges.push_back(r);
    }

    if (ranges.size() < 2) {
      return;
    }

    sensor_msgs::msg::LaserScan msg;
    msg.header.stamp = now();
    msg.header.frame_id = frame_;

    msg.angle_min = static_cast<float>(amin);
    msg.angle_max = static_cast<float>(amax);
    msg.angle_increment =
      static_cast<float>((amax - amin) / static_cast<double>(ranges.size() - 1));

    msg.time_increment = 0.0f;
    msg.scan_time = 0.0f;
    msg.range_min = static_cast<float>(rmin_);
    msg.range_max = static_cast<float>(rmax_);
    msg.ranges = std::move(ranges);
    msg.intensities.clear();

    pub_->publish(std::move(msg));
  }

  // params
  std::string port_;
  std::string frame_;
  int baud_{};
  double amin_deg_{};
  double amax_deg_{};
  double rmin_{};
  double rmax_{};

  // state
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  CYdLidar laser_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<GS2Node>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "Fatal: %s\n", e.what());
  }

  rclcpp::shutdown();
  return 0;
}
