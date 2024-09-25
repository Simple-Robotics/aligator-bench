#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/cost-abstract.hpp>
#include <aligator/modelling/dynamics/context.hpp>
#include <aligator/modelling/dynamics/ode-abstract.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>

namespace alcontext = aligator::context;
using alcontext::ODEAbstract;
using alcontext::ODEData;
using alcontext::TrajOptProblem;
using VectorSpace = proxsuite::nlp::VectorSpaceTpl<double>;

const VectorSpace state_space{2};
using QuadStateCost = aligator::QuadraticStateCostTpl<double>;

struct PendulumDynamics : ODEAbstract {
  ALIGATOR_DYNAMIC_TYPEDEFS(double);
  PendulumDynamics(double mass, double length)
      : ODEAbstract(state_space, 1), mass_(mass), length_(length),
        gravity_(9.81) {}
  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ODEData &data) const {
    const double puls2 = gravity_ / length_;
    const double sn = std::sin(x[0]);
    const double ddth = u[0] - puls2 * sn;

    data.xdot_[0] = x[1];
    data.xdot_[1] = ddth;
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &,
                ODEData &data) const {
    const double cn = std::cos(x[0]);
    const double puls2 = gravity_ / length_;
    data.Jx_.row(0) << 0., 1.;
    data.Jx_.row(1) << -puls2 * cn, 0.;

    data.Ju_.col(0) << 0., 1.;
  }

  double mass_;
  double length_;
  double gravity_;
};

struct PendulumConfig {
  double mass;
  double length;
  uint nsteps;
};

TrajOptProblem createPendulumProblem(PendulumConfig cfg) {
  using namespace aligator::context;
  const int nu = 1;
  const int nx = state_space.nx();
  VectorXs x0 = state_space.neutral();
  QuadStateCost term_cost{state_space, nu, x0, MatrixXs::Identity(nx, nx)};
  TrajOptProblem problem{x0, 1, state_space, term_cost};

  for (uint i = 0; i < cfg.nsteps; i++) {
  }

  return problem;
}

int main() {
  auto problem = createPendulumProblem({1.0, 1.0, 40});
  return 0;
}
