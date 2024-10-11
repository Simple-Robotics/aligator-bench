#include "linear-problem.hpp"
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/costs/quad-costs.hpp>
#include <aligator/modelling/linear-discrete-dynamics.hpp>
#include <aligator/modelling/linear-function.hpp>

#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>

auto createLinearProblem(const size_t horizon, const int nx, const int nu,
                         bool terminal) -> TrajOptProblem {
  using aligator::LinearFunctionTpl;
  using aligator::QuadraticCostTpl;
  using aligator::dynamics::LinearDiscreteDynamicsTpl;
  MatrixXs w_x{nx, nx};
  w_x.setIdentity();
  MatrixXs w_u{nu, nu};
  w_u.setIdentity();

  MatrixXs A = MatrixXs::Ones(nx, nx);
  MatrixXs B = MatrixXs::Identity(nx, nu);

  QuadraticCostTpl<double> cost{w_x, w_u};
  LinearDiscreteDynamicsTpl<double> ddyn{A, B, VectorXs::Zero(nx)};
  StageModel stage{cost, ddyn};
  std::vector<xyz::polymorphic<StageModel>> stages{horizon, stage};

  VectorXs x0 = VectorXs::Random(nx);
  TrajOptProblem problem{x0, std::move(stages), cost};

  if (terminal) {
    LinearFunctionTpl<double> tfunc{nx, nu, 1};
    tfunc.A_.setOnes();
    tfunc.B_.setZero();
    tfunc.d_.setZero();
    problem.addTerminalConstraint(
        tfunc, proxsuite::nlp::EqualityConstraintTpl<double>{});
  }
  return problem;
}
