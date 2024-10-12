#include <aligator/core/constraint.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/modelling/costs/quad-costs.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/dynamics/multibody-free-fwd.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include "ur5-util.hpp"

namespace pin = pinocchio;
using alcontext::MatrixXs;
using alcontext::VectorXs;

xyz::polymorphic<alcontext::StageFunction>
createUr5EeResidual(Space space, Eigen::Vector3d ee_pos) {
  using aligator::FrameTranslationResidualTpl;
  const auto frame_id = space.getModel().getFrameId("tool0");
  const auto &model = space.getModel();
  const auto ndx = space.ndx();
  const auto nu = model.nv;
  return FrameTranslationResidualTpl<double>{ndx, nu, model, ee_pos, frame_id};
}

aligator::CostStackTpl<double> createUr5Cost(Space space, double dt) {
  const pin::Model &model = space.getModel();
  const auto nu = model.nv;
  const auto ndx = space.ndx();
  aligator::CostStackTpl<double> costs{space, nu};

  MatrixXs Wx;
  Wx.setIdentity(ndx, ndx);
  Wx *= 0.1;

  MatrixXs Wu{nu, nu};
  Wu.setIdentity();
  Wu *= 0.1;

  costs.addCost("quad", aligator::QuadraticCostTpl<double>{Wx, Wu}, dt);
  return costs;
}

aligator::dynamics::IntegratorSemiImplEulerTpl<double>
createDynamics(Space space, double dt) {
  using namespace aligator::dynamics;
  MultibodyFreeFwdDynamicsTpl<double> ode{space};
  IntegratorSemiImplEulerTpl<double> integ{ode, dt};
  return integ;
}

xyz::polymorphic<alcontext::CostAbstract>
createTerminalCost(Space space, Eigen::Vector3d ee_pos) {
  auto res = createUr5EeResidual(space, ee_pos);
  Eigen::Matrix3d wr = Eigen::Matrix3d::Identity();
  return aligator::QuadraticResidualCostTpl<double>{space, std::move(res), wr};
}

xyz::polymorphic<alcontext::CostAbstract> createRegTerminalCost(Space space) {
  auto x0 = space.neutral();
  const auto ndx = space.ndx();
  const auto nu = space.getModel().nv;
  MatrixXs wr = 1e-3 * MatrixXs::Identity(ndx, ndx);
  return aligator::QuadraticStateCostTpl<double>{space, nu, x0, wr};
}
