#include "ipopt-solver.hpp"
#include "ipopt-interface.hpp"

#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>

namespace aligator_bench {

SolverIpopt::SolverIpopt(bool rethrow_nonipopt_exception) {
  ipopt_app_ = std::make_unique<Ipopt::IpoptApplication>();

  ipopt_app_->RethrowNonIpoptException(rethrow_nonipopt_exception);
}

Ipopt::ApplicationReturnStatus SolverIpopt::setup(const TrajOptProblem &problem,
                                                  bool verbose) {
  adapter_ = new TrajOptIpoptNLP(problem, verbose);
  Ipopt::ApplicationReturnStatus status;
  status = ipopt_app_->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    fmt::println("\n ** Error during initialization! (Code {:d})", int(status));
  }

  setOption("jacobian_approximation", "exact");
  setOption("print_level", 4);
  setOption("linear_solver", "mumps");

  return status;
}

void SolverIpopt::setOption(const std::string &name, const std::string &value) {
  ipopt_app_->Options()->SetStringValue(name, value);
}
void SolverIpopt::setOption(const std::string &name, int value) {
  ipopt_app_->Options()->SetIntegerValue(name, value);
}
void SolverIpopt::setOption(const std::string &name, double value) {
  ipopt_app_->Options()->SetNumericValue(name, value);
}

Ipopt::ApplicationReturnStatus SolverIpopt::solve() {
  auto status = ipopt_app_->OptimizeTNLP(adapter_);
  if (status != Ipopt::Solve_Succeeded) {
    fmt::println("\n ** Error during optimization! (Code {:d})", int(status));
  }
  return status;
}

} // namespace aligator_bench
