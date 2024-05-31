#pragma once

#include <string>

namespace aligator_bench {

extern const std::string PACKAGE_DIRS_BASE;

struct robot_spec {
  std::string urdfPath;
  std::string srdfPath;
  bool floatingBase;
};

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key);

} // namespace aligator_bench
