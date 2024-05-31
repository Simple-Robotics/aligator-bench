#pragma once

#include "robot_spec.hpp"

#include <pinocchio/multibody/model.hpp>

namespace aligator_bench {

void loadModelFromSpec(const robot_spec &spec, pinocchio::Model &model);

inline void loadModelFromToml(const std::string &tomlFile,
                              const std::string &key, pinocchio::Model &model) {
  robot_spec spec = loadRobotSpecFromToml(tomlFile, key);
  loadModelFromSpec(spec, model);
}

inline pinocchio::Model loadModelFromToml(const std::string &tomlFile,
                                          const std::string &key) {
  pinocchio::Model model;
  robot_spec spec = loadRobotSpecFromToml(tomlFile, key);
  loadModelFromSpec(spec, model);
  return model;
}

} // namespace aligator_bench
