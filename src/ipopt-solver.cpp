#include "ipopt-solver.hpp"

#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>

namespace aligator_bench {

SolverIpopt::SolverIpopt(bool rethrow_nonipopt_exception) {
  ipopt_app_ = std::make_unique<Ipopt::IpoptApplication>();

  ipopt_app_->RethrowNonIpoptException(rethrow_nonipopt_exception);
}

void SolverIpopt::setup(const TrajOptProblem &problem) {}

} // namespace aligator_bench
