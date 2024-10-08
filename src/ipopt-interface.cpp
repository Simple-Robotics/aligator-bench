#include "./types.hpp"
#include "ipopt-interface.hpp"

#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/stage-data.hpp>
#include <aligator/core/cost-abstract.hpp>

#define ALIBENCH_ASSERT_PRETTY(expr, msg)                                      \
  if (!(expr)) {                                                               \
    ALIGATOR_RUNTIME_ERROR(msg);                                               \
  }

namespace aligator_bench {

TrajOptIpoptNLP::TrajOptIpoptNLP(const TrajOptProblem &problem)
    : Ipopt::TNLP(), problem_(problem), problem_data_(problem) {
  const size_t nsteps = problem_.numSteps();
  xs_.resize(nsteps + 1);
  us_.resize(nsteps);
  const auto &stages = problem.stages_;
  nvars_ = 0;
  nconstraints_ = 0;
  idx_xu_.resize(nsteps + 1);
  idx_constraints_.reserve(nsteps + 2);
  idx_constraints_[0] = 0;
  nconstraints_ += problem_.init_constraint_->nr;

  for (size_t i = 0; i < nsteps; i++) {
    const int nxi = stages[i]->nx1();
    const int ndxi = stages[i]->ndx1();
    const int nui = stages[i]->nu();
    xs_[i].setZero(nxi);
    us_[i].setZero(nui);
    idx_xu_[i] = nvars_;
    idx_constraints_[i + 1] = nconstraints_;

    nvars_ += ndxi + nui;
    nconstraints_ += stages[i]->numDual();
  }
  const int nxN = stages.back()->nx2();
  const int ndxN = stages.back()->ndx2();
  xs_.back().setZero(nxN);
  idx_xu_[nsteps] = nvars_;
  idx_constraints_[nsteps + 1] = nconstraints_;
  nvars_ += ndxN;
  nconstraints_ += problem_.term_cstrs_.totalDim();

  const size_t num_digits =
      std::max(size_t(std::log10(nsteps)) + 1, std::size("Index"));
  const size_t ndigits_vars =
      std::max(size_t(std::log10(nvars_)) + 1, std::size("(x,u)"));

  fmt::println("{:s} indices:", __FUNCTION__);
  fmt::println("Initial constraint from {} to {}", 0, idx_constraints_[1]);

  fmt::print("┌{0:─^{1}}┬{0:─^{2}}┬{0:─^{2}}┐\n", "", num_digits + 1,
             ndigits_vars + 1);
  fmt::print("│ {0:<{1}s}│ {2:^{4}s}│ {3:^{4}s}│\n", "Index", num_digits,
             "(x,u)", "(f,g)", ndigits_vars);

  for (size_t i = 0; i <= nsteps; i++) {
    fmt::print("│ {0:<{1}d}│ {2:<{4}d}│ {3:<{4}d}│\n", i, num_digits,
               idx_xu_[i], idx_constraints_[i + 1], ndigits_vars);
  }
  fmt::print("└{0:─^{1}}┴{0:─^{2}}┴{0:─^{2}}┘\n", "", num_digits + 1,
             ndigits_vars + 1);
  fmt::println("Total number of: variables    {:d}\n"
               "                 constraints  {:d}\n",
               nvars_, nconstraints_);
}

bool TrajOptIpoptNLP::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g,
                                   Index &nnz_h_lag,
                                   IndexStyleEnum &index_style) {
  index_style = C_STYLE;
  if (!problem_.checkIntegrity())
    return false;

  // initial constraint
  {
    const int nr = problem_.init_constraint_->nr;
    const int ndx1 = problem_.init_constraint_->ndx1;
    nnz_jac_g += nr * ndx1;
  }

  for (const auto &stage : problem_.stages_) {
    // dynamics
    const int ndx2 = stage->ndx2();
    nnz_jac_g += ndx2 * (stage->ndx1() + stage->nu() + stage->ndx2());
    // constraints
    const int ndx_nu = stage->ndx1() + stage->nu();
    const int nc = stage->nc();
    nnz_jac_g += nc * ndx_nu;
    // hessian
    nnz_h_lag += ndx_nu * ndx_nu;
  }

  // terminal
  {
    const int ndxN = problem_.term_cost_->ndx();
    nnz_h_lag += ndxN * ndxN;

    auto &constraints = problem_.term_cstrs_;
    nnz_jac_g += int(constraints.totalDim()) * ndxN;
  }

  n = nvars_;
  m = nconstraints_;
  return true;
}

bool TrajOptIpoptNLP::get_bounds_info(Index n, double *x_l, double *x_u,
                                      Index m, double *g_l, double *g_u) {

  ALIBENCH_ASSERT_PRETTY(n == nvars_,
                         "n should be equal to number of primal variables!");
  ALIBENCH_ASSERT_PRETTY(m == nconstraints_,
                         "m should be equal to number of constraints!");
  // 1. variable bounds
  // NOTE: Aligator has no explicit variable bounds
  std::fill_n(x_l, n, -2e19);
  std::fill_n(x_u, n, +2e19);

  // 2. constraint bounds
  // TODO: use actual problem bounds, right now everything is equality

  std::fill_n(g_l, m, 0.);
  std::fill_n(g_u, m, 0.);

  return true;
}

bool TrajOptIpoptNLP::get_starting_point(Index n, bool init_traj, double *traj,
                                         bool init_z, double *z_L, double *z_U,
                                         Index m, bool init_lambda,
                                         double *lambda) {
  ALIBENCH_ASSERT_PRETTY(n == nvars_,
                         "n should be equal to number of primal variables!");
  ALIBENCH_ASSERT_PRETTY(m == nconstraints_,
                         "m should be equal to number of constraints!");
  assert(init_traj == true);
  assert(init_z == false);
  (void)init_traj;
  (void)init_z;
  (void)z_L;
  (void)z_U;

  const auto &stages = problem_.stages_;
  const std::size_t nsteps = problem_.numSteps();

  // std::fill_n(z_L, m, -2e19);
  // std::fill_n(z_U, m, +2e19);

  {
    const int nr = problem_.init_constraint_->nr;
    VecMap{lambda, nr} = lams_[0];
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    // const int nxi = stages[i]->nx1();
    const int ndxi = stages[i]->ndx1();
    const int nui = stages[i]->nu();
    const int ndxip1 = stages[i]->ndx2();
    const int nci = stages[i]->nc();

    const int sidx = idx_xu_[i];
    const int cidx = idx_constraints_[i + 1];

    assert(ndxi == xs_[i].size());
    assert(nui == us_[i].size());
    VecMap{traj + sidx, ndxi} = xs_[i];
    VecMap{traj + sidx + ndxi, nui} = us_[i];

    if (init_lambda) {
      VecMap{lambda + cidx, nci} = lams_[i + 1];
      VecMap{lambda + cidx + nci, ndxip1} = vs_[i];
    }
  }

  {
    // terminal
    const int ndx = problem_.term_cost_->ndx();
    const int nc = int(problem_.term_cstrs_.totalDim());
    const int sidx = idx_xu_[nsteps];
    const int cidx = idx_constraints_[nsteps + 1];
    VecMap{traj + sidx, ndx} = xs_[nsteps];
    VecMap{lambda + cidx, nc} = vs_[nsteps];
  }

  return true;
}

void TrajOptIpoptNLP::update_internal_primal_variables(const double *traj) {
  const auto &stages = problem_.stages_;
  const std::size_t nsteps = problem_.numSteps();
  // 1. turn x vector into controls and states
  for (std::size_t i = 0; i < nsteps; i++) {
    // const int nxi = stages[i]->nx1();
    const int ndxi = stages[i]->ndx1();
    const int nui = stages[i]->nu();
    const int sidx = idx_xu_[i];

    assert(xs_[i].size() == nxi);
    assert(us_[i].size() == nui);
    xs_[i] = ConstVecMap{traj + sidx, ndxi};
    us_[i] = ConstVecMap{traj + sidx + ndxi, nui};
  }
}

void TrajOptIpoptNLP::update_internal_dual_variables(const double *lambda) {
  const std::size_t nsteps = problem_.numSteps();
  const auto &stages = problem_.stages_;
  {
    // initial constraint
    const int nr = problem_.init_constraint_->nr;
    lams_[0] = ConstVecMap{lambda, nr};
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    // const int nxi = stages[i]->nx1();
    const int ndxip1 = stages[i]->ndx2();
    const int nci = stages[i]->nc();

    const int cidx = idx_constraints_[i + 1];

    lams_[i + 1] = ConstVecMap{lambda + cidx, nci};
    vs_[i] = ConstVecMap{lambda + cidx + nci, ndxip1};
  }

  {
    // terminal
    const int nc = int(problem_.term_cstrs_.totalDim());
    const int cidx = idx_constraints_[nsteps + 1];
    vs_[nsteps] = ConstVecMap{lambda + cidx, nc};
  }
}

bool TrajOptIpoptNLP::eval_f(Index n, const double *traj, bool new_x,
                             double &obj_value) {
  ALIBENCH_ASSERT_PRETTY(n == nvars_, "");
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.evaluate(xs_, us_, problem_data_);
  }
  obj_value = problem_data_.cost_;
  return true;
}

bool TrajOptIpoptNLP::eval_grad_f(Index n, const double *traj, bool new_x,
                                  double *grad_f) {
  ALIBENCH_ASSERT_PRETTY(n == nvars_, "");
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.computeDerivatives(xs_, us_, problem_data_);
  }
  const std::size_t nsteps = problem_.numSteps();
  const auto &stages = problem_.stages_;

  for (std::size_t i = 0; i < nsteps; i++) {
    const int ndxi = stages[i]->ndx1();
    const int nui = stages[i]->nu();
    const int sidx = idx_xu_[i];
    auto &sd = problem_data_.stage_data[i];
    VecMap gx{grad_f + sidx, ndxi};
    VecMap gu{grad_f + sidx + ndxi, nui};
    gx = sd->cost_data->Lx_;
    gu = sd->cost_data->Lu_;
  }

  const int ndxN = problem_.term_cost_->ndx();
  VecMap gxN{grad_f + idx_xu_[nsteps], ndxN};
  gxN = problem_data_.term_cost_data->Lx_;

  return true;
}

bool TrajOptIpoptNLP::eval_g(Index, const double *traj, bool new_x, Index,
                             double *g) {
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.evaluate(xs_, us_, problem_data_);
  }

  {
    // initial
    const int nr = problem_.init_constraint_->nr;
    VecMap{g, nr} = problem_data_.init_data->value_;
  }

  const std::size_t nsteps = problem_.numSteps();
  const auto &stages = problem_.stages_;
  for (std::size_t i = 0; i < nsteps; i++) {
    const int nc = stages[i]->nc();
    const int ndx2 = stages[i]->ndx2();

    const int cidx = idx_constraints_[i + 1];
    const auto &sd = problem_data_.stage_data[i];
    VecMap{g + cidx, ndx2} = sd->dynamics_data->value_;
    // TODO: fix for multiple constraints...
    VecMap{g + cidx + ndx2, nc} = sd->constraint_data[0]->value_;
  }

  {
    // terminal
    const int nc = int(problem_.term_cstrs_.totalDim());
    const int cidx = idx_constraints_[nsteps + 1];
    // TODO: fix for multiple constraints
    VecMap{g + cidx, nc} = problem_data_.term_cstr_data[0]->value_;
  }

  return true;
}

bool TrajOptIpoptNLP::eval_jac_g(Index, const double *traj, bool new_x, Index,
                                 Index nele_jac, Index *iRow, Index *jCol,
                                 double *values) {
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.computeDerivatives(xs_, us_, problem_data_);
  }
  return true;
}

void TrajOptIpoptNLP::finalize_solution(
    SolverReturn status, Index n, const double *x, const double *z_L,
    const double *z_U, Index m, const double *g, const double *lambda,
    double obj_value, const IpoptData *ip_data,
    Ipopt::IpoptCalculatedQuantities *ip_cq) {}

} // namespace aligator_bench
