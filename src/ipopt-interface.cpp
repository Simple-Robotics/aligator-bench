#include "types.hpp"
#include "triang_util.hpp"
#include "ipopt-interface.hpp"

#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/stage-data.hpp>
#include <aligator/core/cost-abstract.hpp>

#include <boost/unordered_map.hpp>
#include <proxsuite-nlp/modelling/constraints.hpp>

#define ALIBENCH_ASSERT_PRETTY(expr, ...)                                      \
  if (!(expr)) {                                                               \
    ALIGATOR_RUNTIME_ERROR(__VA_ARGS__);                                       \
  }

namespace aligator_bench {

TrajOptIpoptNLP::TrajOptIpoptNLP(const TrajOptProblem &problem, bool verbose)
    : Ipopt::TNLP(), problem_(problem), problem_data_(problem),
      verbose_(verbose) {
  const size_t nsteps = problem_.numSteps();
  xs_.resize(nsteps + 1);
  us_.resize(nsteps);
  lams_.resize(nsteps + 1);
  vs_.resize(nsteps + 1);
  const auto &stages = problem.stages_;
  nvars_ = 0;
  nconstraints_ = 0;
  idx_xu_.resize(nsteps + 1);
  idx_constraints_.reserve(nsteps + 2);
  idx_constraints_[0] = 0;
  nconstraints_ += problem_.init_constraint_->nr;

  {
    // initial
    lams_[0].setZero(problem_.init_constraint_->nr);
  }

  for (size_t i = 0; i < nsteps; i++) {
    const int nxi = stages[i]->nx1();
    const int ndxi = stages[i]->ndx1();
    const int nui = stages[i]->nu();
    const int nc = stages[i]->nc();
    const int ndx2 = stages[i]->ndx2();
    xs_[i].setZero(nxi);
    us_[i].setZero(nui);
    lams_[i + 1].setZero(ndx2);
    vs_[i].setZero(nc);
    idx_xu_[i] = nvars_;
    idx_constraints_[i + 1] = nconstraints_;

    nvars_ += ndxi + nui;
    nconstraints_ += stages[i]->numDual();
  }
  {
    // terminal
    const int nxN = stages.back()->nx2();
    const int ndxN = stages.back()->ndx2();
    const int ncN = int(problem_.term_cstrs_.totalDim());
    xs_[nsteps].setZero(nxN);
    vs_[nsteps].setZero(ncN);
    idx_xu_[nsteps] = nvars_;
    idx_constraints_[nsteps + 1] = nconstraints_;
    nvars_ += ndxN;
    nconstraints_ += problem_.term_cstrs_.totalDim();
  }

  if (verbose_) {
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
}

bool TrajOptIpoptNLP::get_nlp_info(Index &n, Index &m, Index &nnz_jac_g,
                                   Index &nnz_h_lag,
                                   IndexStyleEnum &index_style) {
  nnz_jac_g = 0;
  // NOTE: only the lower diagonal entries
  nnz_h_lag = 0;
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
    const int ndx1 = stage->ndx1();
    const int nu = stage->nu();
    const int ndx2 = stage->ndx2();
    const int nc = stage->nc();
    nnz_jac_g += ndx2 * (ndx1 + nu + ndx2);
    // constraints
    const int ndx_nu = ndx1 + nu;
    nnz_jac_g += nc * ndx_nu;
    // hessian
    nnz_h_lag += ndx_nu * (ndx_nu + 1) / 2;
  }

  // terminal
  {
    const int ndxN = problem_.term_cost_->ndx();
    nnz_h_lag += ndxN * (ndxN + 1) / 2;

    auto &constraints = problem_.term_cstrs_;
    nnz_jac_g += int(constraints.totalDim()) * ndxN;
  }

  n = nvars_;
  m = nconstraints_;
  return true;
}

using alcontext::ConstraintSet;
using ZeroSet = proxsuite::nlp::EqualityConstraintTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;

void bounds_equality(int m, double *g_l, double *g_u, void * = NULL) {
  std::fill_n(g_l, m, 0.);
  std::fill_n(g_u, m, 0.);
}
void bounds_negative(int m, double *g_l, double *g_u, void *) {
  std::fill_n(g_l, m, -2e19);
  std::fill_n(g_u, m, 0.);
}
void bounds_box(int m, double *g_l, double *g_u, void *set_) {
  const BoxConstraint *set = reinterpret_cast<const BoxConstraint *>(set_);
  VecMap{g_l, m} = set->lower_limit;
  VecMap{g_u, m} = set->upper_limit;
}

typedef void (*bounds_dispatch_t)(int, double *, double *, void *);
static boost::unordered_map<std::type_index, bounds_dispatch_t>
    __aligatorConstraintDispatch{{typeid(ZeroSet), bounds_equality},
                                 {typeid(NegativeOrthant), bounds_negative},
                                 {typeid(BoxConstraint), bounds_box}};

void callConstraintDispatch(int nr, double *g_l, double *g_u,
                            const ConstraintSet &set) {
  return __aligatorConstraintDispatch[typeid(set)](nr, g_l, g_u, (void *)&set);
}

bool TrajOptIpoptNLP::get_bounds_info(Index n, double *x_l, double *x_u,
                                      Index m, double *g_l, double *g_u) {

  ALIBENCH_ASSERT_PRETTY(n == nvars_,
                         "n should be equal to number of primal variables!");
  ALIBENCH_ASSERT_PRETTY(m == nconstraints_,
                         "m should be equal to number of constraints!");
  const std::size_t nsteps = problem_.numSteps();
  const auto &sds = problem_.stages_;

  // 1. variable bounds
  // NOTE: Aligator has no explicit variable bounds
  std::fill_n(x_l, n, -2e19);
  std::fill_n(x_u, n, +2e19);

  // 2. constraint bounds
  // initialize all bounds to zero

  std::fill_n(g_l, m, 0.);
  std::fill_n(g_u, m, 0.);

  const int nr0 = problem_.init_constraint_->nr;
  bounds_equality(nr0, g_l, g_u);

  for (std::size_t i = 0; i < nsteps; i++) {
    const auto &stage = *sds[i];
    const int ndx2 = stage.ndx2();
    int cidx = idx_constraints_[i + 1];

    cidx += ndx2; // dynamic constraints are ok
    for (std::size_t j = 0; j < stage.numConstraints(); j++) {
      const int nrj = stage.constraints_.funcs[j]->nr;
      const ConstraintSet &set = *stage.constraints_.sets[j];
      callConstraintDispatch(nrj, g_l + cidx, g_u + cidx, set);
      cidx += nrj;
    }
  }

  const std::size_t nm_c = problem_.term_cstrs_.size();
  int cidx = idx_constraints_[nsteps + 1];
  for (std::size_t j = 0; j < nm_c; j++) {
    const int nrj = problem_.term_cstrs_.funcs[j]->nr;
    const ConstraintSet &set = *problem_.term_cstrs_.sets[j];
    callConstraintDispatch(nrj, g_l + cidx, g_u + cidx, set);
    cidx += nrj;
  }

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

  if (init_lambda) {
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
      VecMap{lambda + cidx, ndxip1} = lams_[i + 1];
      VecMap{lambda + cidx + ndxip1, nci} = vs_[i];
    }
  }

  {
    // terminal
    const int ndx = problem_.term_cost_->ndx();
    const int nc = int(problem_.term_cstrs_.totalDim());
    const int sidx = idx_xu_[nsteps];
    const int cidx = idx_constraints_[nsteps + 1];
    VecMap{traj + sidx, ndx} = xs_[nsteps];
    if (init_lambda)
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

    assert(xs_[i].size() == ndxi);
    assert(us_[i].size() == nui);
    xs_[i] = ConstVecMap{traj + sidx, ndxi};
    us_[i] = ConstVecMap{traj + sidx + ndxi, nui};
  }

  const int ndxN = problem_.term_cost_->ndx();
  xs_[nsteps] = ConstVecMap{traj + idx_xu_[nsteps], ndxN};
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
    problem_.computeDerivatives(xs_, us_, problem_data_);
  }
  obj_value = problem_data_.cost_;
  return true;
}

bool TrajOptIpoptNLP::eval_grad_f(Index n, const double *traj, bool new_x,
                                  double *grad_f) {
  ALIBENCH_ASSERT_PRETTY(n == nvars_, "");
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.evaluate(xs_, us_, problem_data_);
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

bool TrajOptIpoptNLP::eval_g(Index, const double *traj, bool new_x, Index m,
                             double *g) {
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.evaluate(xs_, us_, problem_data_);
    problem_.computeDerivatives(xs_, us_, problem_data_);
  }

  ALIBENCH_ASSERT_PRETTY(m == nconstraints_, "m != nconstraints");
  std::fill_n(g, m, 0.);

  {
    // initial
    const int nr = problem_.init_constraint_->nr;
    VecMap{g, nr} = problem_data_.init_data->value_;
  }

  const std::size_t nsteps = problem_.numSteps();
  const auto &stages = problem_.stages_;
  for (std::size_t i = 0; i < nsteps; i++) {
    const int ndx2 = stages[i]->ndx2();

    int cidx = idx_constraints_[i + 1];
    const auto &sd = problem_data_.stage_data[i];
    VecMap{g + cidx, ndx2} = sd->dynamics_data->value_;
    cidx += ndx2;
    for (size_t k = 0; k < sd->constraint_data.size(); k++) {
      const int nr = sd->constraint_data[k]->nr;
      VecMap{g + cidx, nr} = sd->constraint_data[k]->value_;
      cidx += nr;
    }
  }

  if (!problem_.term_cstrs_.empty()) {
    // terminal
    auto &tcsd = problem_data_.term_cstr_data[0];
    const int nc = int(problem_.term_cstrs_.totalDim());
    const int cidx = idx_constraints_[nsteps + 1];
    assert(nc > 0);
    assert(nconstraints_ == cidx + nc);
    // TODO: fix for multiple constraints
    VecMap{g + cidx, nc} = tcsd->value_;
  }

  return true;
}

bool TrajOptIpoptNLP::eval_jac_g(Index, const double *traj, bool new_x, Index,
                                 Index nele_jac, Index *iRow, Index *jCol,
                                 double *values) {
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.evaluate(xs_, us_, problem_data_);
    problem_.computeDerivatives(xs_, us_, problem_data_);
  }
  const std::size_t nsteps = problem_.numSteps();

  if (values == NULL) {
    std::size_t idx = 0;

    // NOTE: The indices will be filled up in column-major order (inner loop
    // over rows).

    // initial condition
    {
      const int cid = idx_constraints_[0];
      auto &init_cond = problem_.init_constraint_;
      const int ndx = init_cond->ndx1;
      const int nr = init_cond->nr;
      for (int idx_col = 0; idx_col < ndx; idx_col++) {
        for (int idx_row = 0; idx_row < nr; idx_row++) {
          iRow[idx] = cid + idx_row;
          jCol[idx] = idx_col;
          idx++;
        }
      }
    }

    for (std::size_t i = 0; i < nsteps; i++) {
      auto &stage = problem_.stages_[i];
      const int ndx = stage->ndx1();
      const int nu = stage->nu();
      const int ndx2 = stage->ndx2();
      int cid = idx_constraints_[i + 1];

      for (int idx_col = 0; idx_col < ndx + nu + ndx2; idx_col++) {
        for (int idx_row = 0; idx_row < ndx2; idx_row++) {
          iRow[idx] = cid + idx_row;
          jCol[idx] = idx_xu_[i] + idx_col;
          idx++;
        }
      }

      cid += ndx2;
      // treat each constraint separately
      for (size_t k = 0; k < stage->numConstraints(); k++) {
        const int nr = stage->constraints_.funcs[k]->nr;
        // nr rows
        // ndx + nu cols
        for (int idx_col = 0; idx_col < ndx + nu; idx_col++) {
          for (int idx_row = 0; idx_row < nr; idx_row++) {
            iRow[idx] = cid + idx_row;
            jCol[idx] = idx_xu_[i] + idx_col;
            idx++;
          }
        }
        cid += nr;
      }
    }

    {
      // terminal constraint
      const int nr = int(problem_.term_cstrs_.totalDim());
      const int ndx = problem_.term_cost_->ndx();
      const int cid = idx_constraints_[nsteps + 1];
      for (int idx_col = 0; idx_col < ndx; idx_col++) {
        for (int idx_row = 0; idx_row < nr; idx_row++) {
          iRow[idx] = cid + idx_row;
          jCol[idx] = idx_xu_[nsteps] + idx_col;
          idx++;
        }
      }
    }

    ALIBENCH_ASSERT_PRETTY(
        int(idx) == nele_jac,
        "Number of Jacobian elements set ({:d}) does not fit nnz_jac_g ({:d})",
        int(idx), nele_jac);
  } else {
    assert(iRow == NULL);
    assert(jCol == NULL);
    auto &sds = problem_data_.stage_data;
    double *ptr = values;
    std::fill_n(ptr, nele_jac, 0.);

    // initial condition
    {
      auto &idd = problem_data_.init_data;
      const int ndx = idd->ndx1;
      const int nr = idd->nr;
      MatMap jx0{ptr, nr, ndx};
      jx0 = idd->Jx_;
      ptr += ndx * nr;
    }

    for (std::size_t i = 0; i < nsteps; i++) {
      auto &stage = problem_.stages_[i];
      const int ndx = stage->ndx1();
      const int nu = stage->nu();
      const int ndx2 = stage->ndx2();

      MatMap djx{ptr, ndx2, ndx};
      ptr += ndx2 * ndx;
      MatMap dju{ptr, ndx2, nu};
      ptr += ndx2 * nu;
      MatMap djy{ptr, ndx2, ndx2};
      ptr += ndx2 * ndx2;

      djx = sds[i]->dynamics_data->Jx_;
      dju = sds[i]->dynamics_data->Ju_;
      djy = sds[i]->dynamics_data->Jy_;

      // if (nc > 0) {
      //   MatMap jx{ptr, nc, ndx};
      //   ptr += nc * ndx;
      //   MatMap ju{ptr, nc, nu};
      //   ptr += nc * nu;

      //   jx = sds[i]->constraint_data[0]->Jx_;
      //   ju = sds[i]->constraint_data[0]->Ju_;
      // }
      auto &cds = sds[i]->constraint_data;
      for (size_t k = 0; k < cds.size(); k++) {
        const int nr = cds[k]->nr;
        MatMap jx{ptr, nr, ndx};
        ptr += nr * ndx;
        MatMap ju{ptr, nr, nu};
        ptr += nr * nu;

        jx = cds[k]->Jx_;
        ju = cds[k]->Ju_;
      }
    }

    if (!problem_.term_cstrs_.empty()) {
      // terminal constraint
      auto &tcd = problem_data_.term_cstr_data[0];
      const int nr = int(problem_.term_cstrs_.totalDim());
      assert(nr == tcd->nr);
      const int ndx = tcd->ndx1;
      MatMap jx{ptr, nr, ndx};
      jx = tcd->Jx_;
      ptr += nr * ndx;
    }
    auto d = std::distance(values, ptr);
    ALIBENCH_ASSERT_PRETTY(d == nele_jac, "d != nnz_jac_g");
  }
  return true;
}

bool TrajOptIpoptNLP ::eval_h(Index n, const double *traj, bool new_x,
                              double obj_factor, Index m, const double *lambda,
                              bool new_lambda, Index nele_hess, Index *iRow,
                              Index *jCol, double *values) {
  if (new_x) {
    this->update_internal_primal_variables(traj);
    problem_.evaluate(xs_, us_, problem_data_);
    problem_.computeDerivatives(xs_, us_, problem_data_);
  }
  const std::size_t nsteps = problem_.numSteps();
  if (values == NULL) {
    std::size_t idx = 0;
    for (std::size_t i = 0; i < nsteps; i++) {
      const auto &stage = problem_.stages_[i];
      const int ndxi = stage->ndx1();
      const int nui = stage->nu();
      // add (xk, uk) sparsity pattern
      // just the lower triangular part.
      for (int idx_col = 0; idx_col < ndxi + nui; idx_col++) {
        for (int idx_row = idx_col; idx_row < ndxi + nui; idx_row++) {
          iRow[idx] = idx_xu_[i] + idx_row;
          jCol[idx] = idx_xu_[i] + idx_col;
          idx++;
        }
      }
    }
    const int ndxN = problem_.term_cost_->ndx();
    for (int idx_col = 0; idx_col <= ndxN; idx_col++) {
      for (int idx_row = idx_col; idx_row < ndxN; idx_row++) {
        iRow[idx] = idx_xu_[nsteps] + idx_row;
        jCol[idx] = idx_xu_[nsteps] + idx_col;
        idx++;
      }
    }
    ALIBENCH_ASSERT_PRETTY(
        int(idx) == nele_hess,
        "Number of Hessian elements set ({:d}) does not fit nnz_h_lag ({:d})",
        int(idx), nele_hess);
  } else {
    double *ptr = values;
    std::fill_n(ptr, nele_hess, 0.);
    auto &sds = problem_data_.stage_data;
    for (std::size_t i = 0; i < nsteps; i++) {
      const auto &stage = problem_.stages_[i];
      const int ndx = stage->ndx1();
      const int nu = stage->nu();

      auto &H = sds[i]->cost_data->hess_;
      assert(H.cols() == ndx + nu);
      assert(H.rows() == ndx + nu);
      lowTriangAddFromEigen(ptr, H, obj_factor);

      for (size_t k = 0; k < stage->numConstraints(); k++) {
        auto &H = sds[i]->constraint_data[k]->vhp_buffer_;
        assert(H.cols() == ndx + nu);
        assert(H.rows() == ndx + nu);
        lowTriangAddFromEigen(ptr, H, 1.);
      }
      ptr += lowTriangSize(ndx + nu);
    }

    {
      // terminal
      auto &tcd = problem_data_.term_cost_data;
      auto &tcsd = problem_data_.term_cstr_data;
      const int ndx = tcd->ndx_;
      assert(tcd->Lxx_.cols() == ndx);
      assert(tcd->Lxx_.rows() == ndx);
      lowTriangAddFromEigen(ptr, tcd->Lxx_, obj_factor);
      if (!tcsd.empty()) {
        lowTriangAddFromEigen(ptr, tcsd[0]->Hxx_, 1.0);
      }
      ptr += lowTriangSize(ndx);
    }
    const auto d = std::distance(values, ptr);
    ALIBENCH_ASSERT_PRETTY(d == nele_hess,
                           "Final pointer is not the right distance ({:d}, "
                           "expected: {:d}) from the initial one.",
                           d, nele_hess)
  }
  return true;
}

void TrajOptIpoptNLP::finalize_solution(SolverReturn status, Index,
                                        const double *traj, const double *z_L,
                                        const double *z_U, Index,
                                        const double *g, const double *lambda,
                                        double obj_value,
                                        const Ipopt::IpoptData *,
                                        Ipopt::IpoptCalculatedQuantities *) {
  (void)z_L;
  (void)z_U;
  (void)g;
  fmt::println("Optimization finished! Solver return status {:d}\n"
               "Objective value: {:.3e}",
               int(status), obj_value);
  update_internal_primal_variables(traj);
  update_internal_dual_variables(lambda);
}

} // namespace aligator_bench
