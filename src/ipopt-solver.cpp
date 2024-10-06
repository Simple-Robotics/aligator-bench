#include "ipopt-solver.hpp"
#include "ipopt-interface.hpp"

#include <IpIpoptApplication.hpp>
#include <aligator/core/traj-opt-problem.hpp>

namespace aligator_bench {

SolverIpopt::SolverIpopt(bool rethrow_nonipopt_exception) {
  ipopt_app_ = std::make_unique<Ipopt::IpoptApplication>();

  ipopt_app_->RethrowNonIpoptException(rethrow_nonipopt_exception);
}

void SolverIpopt::setup(const TrajOptProblem &problem) {
  // TrajOptIpoptNLP nlp_adapter {problem};
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
