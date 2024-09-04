#pragma once

#include <filesystem>
#include <string>

namespace aligator_bench {
namespace fs = std::filesystem;

extern const fs::path PACKAGE_DIRS_BASE;

struct robot_spec {
  std::string path;
  std::string urdfPath;
  std::string srdfPath;
  std::string refPosture;
  bool floatingBase;
};

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key, bool verbose = false);

} // namespace aligator_bench
