#include "robot_load.hpp"
#include "robot_spec.hpp"
#include <pinocchio/parsers/urdf.hpp>

namespace pinocchio {
namespace urdf {
extern template Model &
buildModel<double, context::Options>(const std::string &filename, Model &model,
                                     const bool verbose);
} // namespace urdf
} // namespace pinocchio

namespace aligator_bench {

void loadModelFromSpec(const robot_spec &spec, pinocchio::Model &model) {
  pinocchio::urdf::buildModel(spec.urdfPath, model);
}

void loadModelFromToml(const std::string &tomlFile, const std::string &key,
                       pinocchio::Model &model) {
  robot_spec spec = loadRobotSpecFromToml(tomlFile, key);
  loadModelFromSpec(spec, model);
}

} // namespace aligator_bench
