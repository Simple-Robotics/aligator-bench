#pragma once

#include <IpTNLP.hpp>
#include <aligator/context.hpp>
#include <aligator/core/traj-opt-data.hpp>

namespace aligator_bench {

namespace alcontext = aligator::context;

namespace {
using alcontext::MatrixXs;
using alcontext::TrajOptData;
using alcontext::TrajOptProblem;
using alcontext::VectorXs;
} // namespace

static_assert(std::is_same_v<aligator::context::Scalar, double>);
static_assert(std::is_same_v<Ipopt::Number, double>);

/// \brief Implement an Ipopt::TNLP class adapting \c aligator::TrajOptProblem
/// instances.
/// Inspired by ETHZ's ifopt: https://github.com/ethz-adrl/ifopt
///
/// \details The layout of this wrapper is as follows: the primal variable
/// is laid out as interleaved states and controls \f$ (x_0, u_0, x_1, ..., x_N)
/// \f$.
///
class TrajOptIpoptNLP final : public Ipopt::TNLP {
public:
  using Index = Ipopt::Index;
  using SolverReturn = Ipopt::SolverReturn;
  using IpoptData = Ipopt::IpoptData;

  TrajOptIpoptNLP(const TrajOptProblem &problem, bool verbose = false);

  bool get_nlp_info(Index &n, Index &m, Index &nnz_jac_g, Index &nnz_h_lag,
                    IndexStyleEnum &index_style) override;

  bool get_bounds_info(Index n, double *x_l, double *x_u, Index m, double *g_l,
                       double *g_u) override;

  bool get_starting_point(Index n, bool init_x, double *x, bool init_z,
                          double *z_L, double *z_U, Index m, bool init_lambda,
                          double *lambda) override;

  void update_internal_primal_variables(const double *traj);
  void update_internal_dual_variables(const double *multipliers);

  bool eval_f(Index n, const double *x, bool new_x, double &obj_value) override;

  bool eval_grad_f(Index n, const double *x, bool new_x,
                   double *grad_f) override;

  bool eval_g(Index n, const double *x, bool new_x, Index m,
              double *g) override;

  bool eval_jac_g(Index n, const double *x, bool new_x, Index m, Index nele_jac,
                  Index *iRow, Index *jCol, double *values) override;

  bool eval_h(Index n, const double *x, bool new_x, double obj_factor, Index m,
              const double *lambda, bool new_lambda, Index nele_hess,
              Index *iRow, Index *jCol, double *values) override;

  void finalize_solution(SolverReturn status, Index n, const double *x,
                         const double *z_L, const double *z_U, Index m,
                         const double *g, const double *lambda,
                         double obj_value, const IpoptData *ip_data,
                         Ipopt::IpoptCalculatedQuantities *ip_cq) override;

  const TrajOptProblem &problem_;
  TrajOptData problem_data_;

  std::vector<VectorXs> xs_;
  std::vector<VectorXs> us_;
  std::vector<VectorXs> lams_;
  std::vector<VectorXs> vs_;

private:
  bool verbose_;
  /// Total number of primal variables in the problem.
  int nvars_;
  /// idx_xu_[i] is the start index of \f$(x_i, u_i) \f$ in the interleaved
  /// state-control vector.
  std::vector<int> idx_xu_;
  /// Number of constraints in the problem.
  int nconstraints_;
  /// idx_constraints_[j] is the start index of the \f$j\f$-th constraint.
  std::vector<int> idx_constraints_;
};

} // namespace aligator_bench
