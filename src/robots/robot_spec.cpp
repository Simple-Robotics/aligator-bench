#include "robot_spec.hpp"
#include <filesystem>
#include <toml/get.hpp>
#include <toml/parser.hpp>

namespace aligator_bench {
namespace fs = std::filesystem;

const std::string PACKAGE_DIRS_BASE =
    fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) / "../..";

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key) {
  const auto data = toml::parse(fname);
  const auto path = fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) /
                    toml::find<std::string>(data, "path");
  const auto urdfSubpath =
      toml::find_or<std::string>(data, "urdf_subpath", "urdf");
  const auto srdfSubpath =
      toml::find_or<std::string>(data, "srdf_subpath", "srdf");

  const auto &node = toml::find(data, key);
  const auto urdfFile = toml::find<std::string>(node, "urdf_filename");
  const auto srdfFile = toml::find_or<std::string>(node, "srdf_filename", "");

  return robot_spec{path / urdfSubpath / urdfFile,
                    path / srdfSubpath / srdfFile,
                    toml::find_or<bool>(data, "free_flyer", false)};
}

} // namespace aligator_bench
