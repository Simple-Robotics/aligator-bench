#pragma once

#include <pinocchio/multibody/model.hpp>

namespace aligator_bench {
struct robot_spec;

void loadModelFromSpec(const robot_spec &spec, pinocchio::Model &model);

void loadModelFromToml(const std::string &tomlFile, const std::string &key,
                       pinocchio::Model &model);

inline pinocchio::Model loadModelFromToml(const std::string &tomlFile,
                                          const std::string &key) {
  pinocchio::Model model;
  loadModelFromToml(tomlFile, key, model);
  return model;
}

} // namespace aligator_bench
