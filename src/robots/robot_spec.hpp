#pragma once

#include <filesystem>
#include <string>

namespace aligator_bench {
namespace fs = std::filesystem;

extern const fs::path PACKAGE_DIRS_BASE;

struct robot_spec {
  std::string urdfPath;
  std::string srdfPath;
  bool floatingBase;
};

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key, bool verbose = false);

} // namespace aligator_bench
