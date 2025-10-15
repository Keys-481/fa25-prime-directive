cmake_minimum_required(VERSION 3.10)
project(gs2_publisher)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Threads REQUIRED)

set(YDLIDAR_PKG_FOUND OFF)
find_package(ydlidar_sdk QUIET)
if(ydlidar_sdk_FOUND)
  set(YDLIDAR_PKG_FOUND ON)
endif()

add_executable(gs2_node src/gs2_node.cpp)
ament_target_dependencies(gs2_node rclcpp sensor_msgs)
target_compile_features(gs2_node PUBLIC cxx_std_17)

if(YDLIDAR_PKG_FOUND)
  message(STATUS "Found ydlidar_sdk package")
  if(TARGET ydlidar_sdk::ydlidar_sdk)
    # PLAIN signature (no PRIVATE/PUBLIC):
    target_link_libraries(gs2_node ydlidar_sdk::ydlidar_sdk Threads::Threads)
  elseif(TARGET ydlidar_sdk)
    target_link_libraries(gs2_node ydlidar_sdk Threads::Threads)
  else()
    message(WARNING "ydlidar_sdk package found, but no target exported; falling back to manual link")
    target_include_directories(gs2_node PRIVATE /usr/local/include /usr/local/include/src)
    target_link_libraries(gs2_node /usr/local/lib/libydlidar_sdk.a Threads::Threads)
  endif()
else()
  message(WARNING "ydlidar_sdk package NOT found; manual include/lib fallback")
  set(YDLIDAR_INCLUDE_DIR "/usr/local/include" CACHE PATH "YDLIDAR include root")
  set(YDLIDAR_INCLUDE_DIR_SRC "/usr/local/include/src" CACHE PATH "YDLIDAR include src dir")
  set(YDLIDAR_LIB "/usr/local/lib/libydlidar_sdk.a" CACHE FILEPATH "YDLIDAR static library")

  target_include_directories(gs2_node PRIVATE ${YDLIDAR_INCLUDE_DIR} ${YDLIDAR_INCLUDE_DIR_SRC})
  target_link_libraries(gs2_node ${YDLIDAR_LIB} Threads::Threads)
endif()

install(TARGETS gs2_node DESTINATION lib/${PROJECT_NAME})
ament_package()

