#include "aligator/core/cost-abstract.hpp"
#include "aligator/core/traj-opt-problem.hpp"

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/dynamics/context.hpp"
#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"
#include "aligator/modelling/dynamics/multibody-free-fwd.hpp"

#include "robots/robot_load.hpp"

namespace alcontext = aligator::context;
namespace pin = pinocchio;
using alcontext::CostAbstract;
using alcontext::StageModel;
using alcontext::TrajOptProblem;
using alcontext::VectorXs;
using xyz::polymorphic;

using aligator_bench::loadRobotSpecFromToml;
using aligator_bench::PACKAGE_DIRS_BASE;
using aligator_bench::robot_spec;

using MultibodyFreeFwd =
    aligator::dynamics::MultibodyFreeFwdDynamicsTpl<double>;
using Space = proxsuite::nlp::MultibodyPhaseSpace<double>;
using aligator::dynamics::IntegratorSemiImplEulerTpl;

auto createCost(Space space) {
  aligator::CostStackTpl<double> costs{space, space.getModel().nv};
  return costs;
}

auto createProblem() {
  const double dt = 1e-2;
  const size_t nsteps = 100;

  pin::Model model = aligator_bench::loadModelFromToml("ur.toml", "ur5");
  std::cout << model << std::endl;

  Space state_space{model};
  const VectorXs x0 = state_space.neutral();

  const MultibodyFreeFwd ode{state_space};
  const IntegratorSemiImplEulerTpl<double> dynamics{ode, dt};
  const auto rcost = createCost(state_space);
  auto tcost = rcost;

  StageModel stage{rcost, dynamics};

  std::vector<polymorphic<StageModel>> stages{nsteps, stage};
  return TrajOptProblem{x0, stages, tcost};
}

int main() {
  auto problem = createProblem();
  return 0;
}
