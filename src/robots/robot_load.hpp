#pragma once

#include <pinocchio/multibody/fwd.hpp>

namespace aligator_bench {
struct robot_spec;

void loadModelFromSpec(const robot_spec &spec, pinocchio::Model &model);

void loadModelFromToml(const std::string &tomlFile, const std::string &key,
                       pinocchio::Model &model);

} // namespace aligator_bench
