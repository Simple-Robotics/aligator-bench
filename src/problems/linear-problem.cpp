#include "linear-problem.hpp"
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/costs/quad-costs.hpp>
#include <aligator/modelling/linear-discrete-dynamics.hpp>
#include <aligator/modelling/linear-function.hpp>

#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>

using EqSet = proxsuite::nlp::EqualityConstraintTpl<double>;

alcontext::TrajOptProblem createLinearProblem(const size_t horizon,
                                              const int nx, const int nu,
                                              bool terminal) {
  using namespace alcontext;
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
  std::vector<xyz::polymorphic<StageModel>> stages;
  for (size_t i = 0; i < horizon; i++) {
    stages.push_back(stage);
    if (horizon > 4 && i == horizon / 2) {
      LinearFunctionTpl<double> sf{nx, nu, 1};
      sf.A_(0, 0) = 1.0;
      sf.d_.setZero();
      stage.addConstraint(sf, EqSet{});

      sf.A_.setZero();
      sf.A_(0, 1) = 2.0;
      stage.addConstraint(sf, EqSet{});
    }
  }

  VectorXs x0 = VectorXs::Random(nx);
  TrajOptProblem problem{x0, std::move(stages), cost};

  if (terminal) {
    LinearFunctionTpl<double> tfunc{nx, nu, 1};
    tfunc.A_.setOnes();
    tfunc.B_.setZero();
    tfunc.d_.setZero();
    problem.addTerminalConstraint(tfunc, EqSet{});
  }
  return problem;
}
