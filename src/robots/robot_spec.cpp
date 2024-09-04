#include "robot_spec.hpp"
#include <filesystem>
#include <iostream>
#include <toml11/find.hpp>
#include <toml11/parser.hpp>

namespace aligator_bench {

const fs::path PACKAGE_DIRS_BASE =
    fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) / "../..";

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key, bool verbose) {
  const auto data = toml::parse(fname);
  const auto path = fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) /
                    toml::find<std::string>(data, "path");
  const auto urdfSubpath =
      toml::find_or<std::string>(data, "urdf_subpath", "urdf");
  const auto srdfSubpath =
      toml::find_or<std::string>(data, "srdf_subpath", "srdf");

  const auto &node = toml::find(data, key);
  const std::string urdfFile = toml::find<std::string>(node, "urdf_filename");
  const std::string srdfFile =
      toml::find_or<std::string>(node, "srdf_filename", "");

  robot_spec result{path / urdfSubpath / urdfFile,
                    path / srdfSubpath / srdfFile,
                    toml::find_or<bool>(data, "free_flyer", false)};
  if (verbose) {
    std::cout << "Loaded robot with URDF file " << result.urdfPath << '\n';
    std::cout << "             with SRDF file " << result.srdfPath << std::endl;
    if (result.floatingBase)
      std::cout << "Robot has floating base." << std::endl;
  }
  return result;
}

} // namespace aligator_bench
