#pragma once

#include <Eigen/Core>
#include <span>

inline std::vector<double> traj_coordinate(std::span<Eigen::VectorXd> states,
                                           Eigen::Index i) {
  std::vector<double> out;
  out.reserve(states.size());
  for (const auto &x : states) {
    assert(i < x.size());
    out.push_back(x[i]);
  }
  return out;
}
