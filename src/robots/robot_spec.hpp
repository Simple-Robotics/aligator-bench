#include <filesystem>
#include <string>

namespace aligator_bench {
namespace fs = std::filesystem;

inline const std::string PACKAGE_DIRS_BASE =
    fs::path(EXAMPLE_ROBOT_DATA_MODEL_DIR) / "../..";

struct robot_spec {
  std::string urdfPath;
  std::string srdfPath;
  bool floatingBase;
};

robot_spec loadRobotSpecFromToml(const std::string &fname,
                                 const std::string &key);

} // namespace aligator_bench
