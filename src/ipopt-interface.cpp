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
  const auto &stages = problem_.stages_;
  const std::size_t nsteps = problem_.numSteps();

  {
  }

  for (std::size_t i = 0; i < nsteps; i++) {
    // const int nxi = stages[i]->nx1();
    const int ndxi = stages[i]->ndx1();
    const int nui = stages[i]->nu();
    const int ndxip1 = stages[i]->ndx2();
    const int nci = stages[i]->nc();

    const int sidx = idx_xu_[i];
    const int cidx = idx_constraints_[i + 1];

    VecMap xi{traj + sidx, ndxi};
    VecMap ui{traj + sidx + ndxi, nui};

    xi = xs_[i];
    ui = us_[i];

    if (init_lambda) {
      VecMap{lambda, cidx, nci + ndxip1}.setZero();
    }
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

    ConstVecMap xi{traj + sidx, ndxi};
    ConstVecMap ui{traj + sidx + ndxi, nui};

    xs_[i] = xi;
    us_[i] = ui;
  }
}

void TrajOptIpoptNLP::update_internal_dual_variables(const double *multpliers) {

}

bool TrajOptIpoptNLP::eval_f(Index n, const double *traj, bool new_x,
                             double &obj_value) {
  ALIBENCH_ASSERT_PRETTY(n == nvars_, "");
  this->update_internal_primal_variables(traj);
  // 2. evaluate
  obj_value = problem_.evaluate(xs_, us_, problem_data_);
  return true;
}

bool TrajOptIpoptNLP::eval_grad_f(Index n, const double *traj, bool new_x,
                                  double *grad_f) {
  ALIBENCH_ASSERT_PRETTY(n == nvars_, "");
  this->update_internal_primal_variables(traj);
  const std::size_t nsteps = problem_.numSteps();
  const auto &stages = problem_.stages_;

  problem_.computeDerivatives(xs_, us_, problem_data_);

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

bool TrajOptIpoptNLP::eval_g(Index n, const double *x, bool new_x, Index m,
                             double *g) {
  return true;
}

bool TrajOptIpoptNLP::eval_jac_g(Index n, const double *x, bool new_x, Index m,
                                 Index nele_jac, Index *iRow, Index *jCol,
                                 double *values) {
  return true;
}

void TrajOptIpoptNLP::finalize_solution(
    SolverReturn status, Index n, const double *x, const double *z_L,
    const double *z_U, Index m, const double *g, const double *lambda,
    double obj_value, const IpoptData *ip_data,
    Ipopt::IpoptCalculatedQuantities *ip_cq) {}

} // namespace aligator_bench
