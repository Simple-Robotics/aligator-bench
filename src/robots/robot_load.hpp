#pragma once

#include <pinocchio/multibody/model.hpp>

namespace aligator_bench {
// fwd decl
struct robot_spec;

void loadModelFromSpec(const robot_spec &spec, pinocchio::Model &model,
                       bool verbose = false);

inline pinocchio::Model loadModelFromSpec(const robot_spec &spec,
                                          bool verbose = false) {
  pinocchio::Model model;
  loadModelFromSpec(spec, model, verbose);
  return model;
}

void loadModelFromToml(const std::string &tomlFile, const std::string &key,
                       pinocchio::Model &model, bool verbose = false);

} // namespace aligator_bench
