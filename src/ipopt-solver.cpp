#include "ipopt-solver.hpp"
#include "ipopt-interface.hpp"

#include <IpIpoptApplication.hpp>
#include <IpIpoptData.hpp>
#include <IpSolveStatistics.hpp>
#include <IpIpoptCalculatedQuantities.hpp>
#include <aligator/core/traj-opt-problem.hpp>

namespace aligator_bench {

SolverIpopt::SolverIpopt(bool rethrow_nonipopt_exception) {
  ipopt_app_ = std::make_unique<Ipopt::IpoptApplication>();

  ipopt_app_->RethrowNonIpoptException(rethrow_nonipopt_exception);

  setOption("jacobian_approximation", "exact");
  setOption("print_level", 4);
  setOption("linear_solver", "mumps");
  setOption("max_iter", 300);
}

Ipopt::ApplicationReturnStatus SolverIpopt::setup(const TrajOptProblem &problem,
                                                  bool verbose) {
  adapter_ = new TrajOptIpoptNLP(problem, verbose);
  Ipopt::ApplicationReturnStatus status;
  status = ipopt_app_->Initialize();
  if (status != Ipopt::Solve_Succeeded) {
    fmt::println("\n ** Error during initialization! (Code {:d})", int(status));
  }

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

  // Collect computed quantities.
  // Check source code of IpoptApplication, in IpIpoptApplication.cpp.
  // These are the same values as the solver printout.

  Ipopt::IpoptData *pip_data = Ipopt::GetRawPtr(ipopt_app_->IpoptDataObject());
  num_iter_ = pip_data->iter_count();

  Ipopt::IpoptCalculatedQuantities *pip_cq =
      Ipopt::GetRawPtr(ipopt_app_->IpoptCQObject());
  assert(pip_cq != NULL);
  using Ipopt::ENormType;
  traj_cost_ = pip_cq->unscaled_curr_f();
  dual_infeas_ = pip_cq->unscaled_curr_dual_infeasibility(ENormType::NORM_MAX);
  cstr_violation_ =
      pip_cq->unscaled_curr_nlp_constraint_violation(ENormType::NORM_MAX);
  complementarity_ =
      pip_cq->unscaled_curr_complementarity(0., ENormType::NORM_MAX);

  return status;
}

double SolverIpopt::totalSolveTime() const {
  return ipopt_app_->Statistics()->TotalWallclockTime();
}

const VectorOfVectors &SolverIpopt::xs() const {
  auto &pi = static_cast<TrajOptIpoptNLP &>(*adapter_);
  return pi.xs_;
}

const VectorOfVectors &SolverIpopt::us() const {
  auto &pi = static_cast<TrajOptIpoptNLP &>(*adapter_);
  return pi.us_;
}

const VectorOfVectors &SolverIpopt::lams() const {
  auto &pi = static_cast<TrajOptIpoptNLP &>(*adapter_);
  return pi.lams_;
}

const VectorOfVectors &SolverIpopt::vs() const {
  auto &pi = static_cast<TrajOptIpoptNLP &>(*adapter_);
  return pi.vs_;
}

void SolverIpopt::setInitialGuess(VectorOfVectors xs, VectorOfVectors us) {
  auto &pi = static_cast<TrajOptIpoptNLP &>(*adapter_);
  pi.xs_ = std::move(xs);
  pi.us_ = std::move(us);
}

void SolverIpopt::setAbsTol(double tol) {
  setOption("tol", tol);
  setOption("dual_inf_tol", tol);
  setOption("constr_viol_tol", tol);
  setOption("compl_inf_tol", tol);
}
} // namespace aligator_bench
