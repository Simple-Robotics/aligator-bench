#include "robot_load.hpp"
#include <pinocchio/parsers/urdf.hpp>

namespace aligator_bench {

void loadModelFromSpec(const robot_spec &spec, pinocchio::Model &model) {
  pinocchio::urdf::buildModel(spec.urdfPath, model);
}

} // namespace aligator_bench
