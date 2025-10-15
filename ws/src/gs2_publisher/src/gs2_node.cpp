#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

// YDLidar SDK (installed to /usr/local/include by your `make install`)
#include <src/CYdLidar.h>
#include <core/common/ydlidar_def.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace std::chrono_literals;

// NOTE: Your SDK installs symbols in the **global** namespace (no ydlidar::),
// so we use CYdLidar, LaserScan, and the LidarProp* / TYPE_* enums unqualified.

static void set_str(CYdLidar& l, int prop, const std::string& s, rclcpp::Logger log) {
  if (!l.setlidaropt(prop, s.c_str(), s.size()))
    RCLCPP_WARN(log, "Failed to set string prop %d", prop);
}
static void set_int(CYdLidar& l, int prop, int v, rclcpp::Logger log) {
  if (!l.setlidaropt(prop, &v, sizeof(int)))
    RCLCPP_WARN(log, "Failed to set int prop %d", prop);
}
static void set_bool(CYdLidar& l, int prop, bool v, rclcpp::Logger log) {
  if (!l.setlidaropt(prop, &v, sizeof(bool)))
    RCLCPP_WARN(log, "Failed to set bool prop %d", prop);
}
static void set_float(CYdLidar& l, int prop, float v, rclcpp::Logger log) {
  if (!l.setlidaropt(prop, &v, sizeof(float)))
    RCLCPP_WARN(log, "Failed to set float prop %d", prop);
}

class GS2Node : public rclcpp::Node {
public:
  GS2Node() : Node("gs2_publisher") {
    // Params (override via --ros-args -p key:=val)
    port_     = declare_parameter<std::string>("port", "/dev/ydlidar"); // use a udev symlink if you created one
    baud_     = declare_parameter<int>("baudrate", 921600);
    frame_    = declare_parameter<std::string>("frame_id", "laser_frame");
    amin_deg_ = declare_parameter<double>("angle_min_deg", -50.0);
    amax_deg_ = declare_parameter<double>("angle_max_deg",  50.0);
    rmin_     = declare_parameter<double>("range_min", 0.025);
    rmax_     = declare_parameter<double>("range_max", 1.0); // gs_test uses 1.0 m

    pub_ = create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);

    // ---- Mirror gs_test configuration exactly ----
    // string props
    set_str(laser_, LidarPropSerialPort, port_, get_logger());
    std::string ignore_array; // empty
    set_str(laser_, LidarPropIgnoreArray, ignore_array, get_logger());

    // int props
    set_int(laser_, LidarPropSerialBaudrate, baud_, get_logger());
    int lidar_type  = TYPE_GS;                 // <-- GS family
    set_int(laser_, LidarPropLidarType, lidar_type, get_logger());
    int device_type = YDLIDAR_TYPE_SERIAL;     // serial mode
    set_int(laser_, LidarPropDeviceType, device_type, get_logger());
    int sample_rate = 4;                       // 4 kHz (per sample)
    set_int(laser_, LidarPropSampleRate, sample_rate, get_logger());
    int abnormal_chk = 4;
    set_int(laser_, LidarPropAbnormalCheckCount, abnormal_chk, get_logger());
    int intensity_bits = 8;
    set_int(laser_, LidarPropIntenstiyBit, intensity_bits, get_logger());

    // bool props
    set_bool(laser_, LidarPropFixedResolution, false, get_logger());
    set_bool(laser_, LidarPropReversion,       false, get_logger());
    set_bool(laser_, LidarPropInverted,        false, get_logger());
    set_bool(laser_, LidarPropAutoReconnect,   true,  get_logger());
    bool isSingleChannel = false;
    set_bool(laser_, LidarPropSingleChannel,   isSingleChannel, get_logger());
    set_bool(laser_, LidarPropIntenstiy,       true,  get_logger());
    set_bool(laser_, LidarPropSupportMotorDtrCtrl, true,  get_logger());
    set_bool(laser_, LidarPropSupportHeartBeat,   false, get_logger());

    // float props
    set_float(laser_, LidarPropMaxAngle,   180.0f, get_logger());
    set_float(laser_, LidarPropMinAngle,  -180.0f, get_logger());
    set_float(laser_, LidarPropMaxRange,  static_cast<float>(rmax_), get_logger()); // 1.0 by default
    set_float(laser_, LidarPropMinRange,  static_cast<float>(rmin_), get_logger()); // 0.025
    float freq = 8.0f;
    set_float(laser_, LidarPropScanFrequency, freq, get_logger());

    laser_.setEnableDebug(true);

    RCLCPP_INFO(get_logger(), "Opening GS2 (TYPE_GS) on %s @ %d", port_.c_str(), baud_);
    if (!laser_.initialize())
      throw std::runtime_error(std::string("initialize() failed: ") + laser_.DescribeError());

    // small settle like the sample effectively gets via prompts
    rclcpp::sleep_for(200ms);

    if (!laser_.turnOn())
      throw std::runtime_error(std::string("turnOn() failed: ") + laser_.DescribeError());

    // fast, light polling loop
    timer_ = create_wall_timer(5ms, [this]{ this->tick(); });
  }

  ~GS2Node() override {
    try { laser_.turnOff(); laser_.disconnecting(); } catch(...) {}
  }

private:
  void tick() {
    LaserScan scan;
    if (!laser_.doProcessSimple(scan)) return;

    // publish only requested window (default ±50°)
    const double amin = amin_deg_ * M_PI / 180.0;
    const double amax = amax_deg_ * M_PI / 180.0;

    std::vector<float> ranges;
    ranges.reserve(scan.points.size());
    for (const auto& p : scan.points) {
      if (p.angle < amin || p.angle > amax) continue;
      float r = static_cast<float>(p.range);
      if (r <= 0.f) r = std::numeric_limits<float>::infinity();
      ranges.push_back(r);
    }
    if (ranges.size() < 2) return;

    sensor_msgs::msg::LaserScan msg;
    msg.header.stamp = now();
    msg.header.frame_id = frame_;
    msg.angle_min = amin;
    msg.angle_max = amax;
    msg.angle_increment = (amax - amin) / double(ranges.size() - 1);
    msg.time_increment = 0.0;
    msg.scan_time = 0.0;
    msg.range_min = rmin_;
    msg.range_max = rmax_;
    msg.ranges = std::move(ranges);
    msg.intensities.clear();

    pub_->publish(std::move(msg));
  }

  // params
  std::string port_, frame_;
  int baud_;
  double amin_deg_, amax_deg_, rmin_, rmax_;

  // state
  rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  CYdLidar laser_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try { rclcpp::spin(std::make_shared<GS2Node>()); }
  catch (const std::exception& e) { fprintf(stderr, "Fatal: %s\n", e.what()); }
  rclcpp::shutdown();
  return 0;
}
