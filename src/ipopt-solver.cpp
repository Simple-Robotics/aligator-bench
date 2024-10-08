#include "ipopt-solver.hpp"
#include "ipopt-interface.hpp"

#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>

namespace aligator_bench {

SolverIpopt::SolverIpopt(bool rethrow_nonipopt_exception) {
  ipopt_app_ = std::make_unique<Ipopt::IpoptApplication>();

  ipopt_app_->RethrowNonIpoptException(rethrow_nonipopt_exception);
}

Ipopt::ApplicationReturnStatus
SolverIpopt::setup(const TrajOptProblem &problem) {
  Ipopt::SmartPtr<TrajOptIpoptNLP> nlp_adapter = new TrajOptIpoptNLP(problem);
  Ipopt::ApplicationReturnStatus status;
  status = ipopt_app_->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    fmt::println("\n ** Error during initialization! (Code {:d})", int(status));
  }
  return status;
}

void SolverIpopt::setOption(const std::string &name, std::string_view value) {
  ipopt_app_->Options()->SetStringValue(name, std::string(value));
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

} // namespace aligator_bench
