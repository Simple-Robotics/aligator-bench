#include "ipopt-solver.hpp"

#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>

namespace aligator_bench {

SolverIpopt::SolverIpopt() {
  ipopt_app_ = std::make_unique<Ipopt::IpoptApplication>();
}

void SolverIpopt::setup(const TrajOptProblem &problem) {}

} // namespace aligator_bench
