#include <aligator/core/traj-opt-problem.hpp>

#include <aligator/modelling/costs/quad-costs.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/multibody-free-fwd.hpp>

#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include "robots/robot_load.hpp"

namespace alcontext = aligator::context;
namespace pin = pinocchio;
using alcontext::CostAbstract;
using alcontext::MatrixXs;
using alcontext::StageModel;
using alcontext::TrajOptProblem;
using alcontext::VectorXs;
using xyz::polymorphic;

using MultibodyFreeFwd =
    aligator::dynamics::MultibodyFreeFwdDynamicsTpl<double>;
using Space = proxsuite::nlp::MultibodyPhaseSpace<double>;
using aligator::dynamics::IntegratorSemiImplEulerTpl;

auto createCost(Space space, double dt) {
  auto &model = space.getModel();
  const auto nu = model.nv;
  const auto ndx = space.ndx();
  aligator::CostStackTpl<double> costs{space, nu};

  MatrixXs Wx{ndx, ndx};
  Wx.setIdentity();

  MatrixXs Wu{nu, nu};
  Wu.setIdentity();
  aligator::QuadraticCostTpl<double> quadcost{Wx, Wu};

  costs.addCost(quadcost, 1e-4 * dt);

  return costs;
}

auto createTerminalCost(Space space, VectorXs xf) {
  const auto nu = space.getModel().nv;
  MatrixXs weights{space.ndx(), space.ndx()};
  weights.setIdentity();
  aligator::QuadraticStateCostTpl<double> quadf{space, nu, xf, weights};
  return quadf;
}

TrajOptProblem createProblem() {
  const double tf = 1.0;
  const double dt = 1e-2;
  const size_t nsteps = size_t(tf / dt);

  pin::Model model;
  aligator_bench::loadModelFromToml("ur.toml", "ur5", model);
  std::cout << model << std::endl;

  Space state_space{model};
  const VectorXs x0 = state_space.neutral();

  const MultibodyFreeFwd ode{state_space};
  const IntegratorSemiImplEulerTpl<double> dynamics{ode, dt};
  const auto rcost = createCost(state_space, dt);

  VectorXs xf = state_space.rand();
  const auto tcost = createTerminalCost(state_space, xf);

  StageModel stage{rcost, dynamics};

  std::vector<polymorphic<StageModel>> stages{nsteps, stage};
  return TrajOptProblem{x0, stages, tcost};
}

int main() {
  auto problem = createProblem();

  const double mu_init = 1e-3;
  alcontext::SolverProxDDP solver{1e-4, mu_init};
  solver.verbose_ = aligator::VERBOSE;
  solver.linear_solver_choice = aligator::LQSolverChoice::SERIAL;

  solver.setup(problem);
  solver.run(problem);

  return 0;
}
