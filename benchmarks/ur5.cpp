#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/costs/quad-costs.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/multibody-free-fwd.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>
#include <aligator/utils/rollout.hpp>

#include "aligator-problem-to-altro.hpp"
#include "util.hpp"
#include "robots/robot_load.hpp"

#include <altro/altro.hpp>
#include <matplot/matplot.h>

namespace pin = pinocchio;
namespace alcontext = aligator::context;
using alcontext::CostAbstract;
using alcontext::MatrixXs;
using alcontext::StageModel;
using alcontext::TrajOptProblem;
using alcontext::VectorXs;
using PolyCost = xyz::polymorphic<CostAbstract>;
using ZeroSet = proxsuite::nlp::EqualityConstraintTpl<double>;

using MultibodyFreeFwd =
    aligator::dynamics::MultibodyFreeFwdDynamicsTpl<double>;
using Space = proxsuite::nlp::MultibodyPhaseSpace<double>;
using aligator::dynamics::IntegratorSemiImplEulerTpl;

aligator::CostStackTpl<double> createUr5Cost(Space space, double dt);

xyz::polymorphic<alcontext::StageFunction>
createUr5EeResidual(Space space, Eigen::Vector3d ee_pos);

auto createTerminalCost(Space space, Eigen::Vector3d ee_pos) {
  auto res = createUr5EeResidual(space, ee_pos);
  Eigen::Matrix3d wr = Eigen::Matrix3d::Identity();
  return aligator::QuadraticResidualCostTpl<double>{space, std::move(res), wr};
}

auto createRegTerminalCost(Space space) {
  auto x0 = space.neutral();
  const auto ndx = space.ndx();
  const auto nu = space.getModel().nv;
  MatrixXs wr = 1e-3 * MatrixXs::Identity(ndx, ndx);
  return aligator::QuadraticStateCostTpl<double>{space, nu, x0, wr};
}

alcontext::StageConstraint createUr5TerminalConstraint(Space space,
                                                       Eigen::Vector3d ee_pos);

int main() {
  const double dt = 5e-2;
  const size_t nsteps = 120;
  const size_t max_iters = 400;
  const double tf = nsteps * dt;
  const std::vector<double> times = matplot::linspace(0, tf, nsteps + 1);

  pin::Model model = aligator_bench::loadModelFromToml("ur.toml", "ur5");
  std::cout << model << std::endl;

  const Space space{model};
  VectorXs x0 = space.neutral();
  x0.head<3>() << 0.1, -0.64, 1.38;

  const IntegratorSemiImplEulerTpl<double> dynamics{MultibodyFreeFwd{space},
                                                    dt};
  const auto rcost = createUr5Cost(space, dt);

  Eigen::Vector3d ee_term_pos{0.5, 0.5, 0.1};
  bool use_term_cstr = true;
  const PolyCost term_cost = use_term_cstr
                                 ? createRegTerminalCost(space)
                                 : createTerminalCost(space, ee_term_pos);

  StageModel stage{rcost, dynamics};
  std::vector<VectorXs> us_init;
  for (size_t i = 0; i < nsteps; i++)
    us_init.emplace_back(VectorXs::Zero(dynamics.nu));
  std::vector<VectorXs> xs_init = aligator::rollout(dynamics, x0, us_init);

  std::vector<xyz::polymorphic<StageModel>> stages{nsteps, stage};
  TrajOptProblem problem{x0, stages, term_cost};

  if (use_term_cstr) {
    auto term_constraint = createUr5TerminalConstraint(space, ee_term_pos);
    problem.addTerminalConstraint(term_constraint.func, term_constraint.set);
    fmt::println("Adding a terminal constraint");
  }

  const double mu_init = 1e-3;
  const double tol = 1e-6;
  alcontext::SolverProxDDP solver{tol, mu_init};
  solver.max_iters = max_iters;
  solver.verbose_ = aligator::VERBOSE;
  solver.linear_solver_choice = aligator::LQSolverChoice::SERIAL;
  solver.bcl_params.dyn_al_scale = 1e-12;
  solver.reg_min = 1e-10;

  solver.setup(problem);
  solver.run(problem, xs_init, us_init);
  fmt::println("{}", solver.results_);

  {
    // plot
    matplot::figure();
    auto ax = matplot::gca();
    ax->hold(true);
    ax->title("Aligator");
    for (long i = 0; i < 6; i++) {
      ax->plot(times, traj_coordinate(solver.results_.xs, i))->line_width(1.5);
    }
    matplot::show();
  }

  {
    using altro::ErrorCodes;
    using altro::SolveStatus;
    altro::ALTROSolver solver =
        std::move(*aligator_bench::initAltroFromAligatorProblem(problem));
    altro::AltroOptions opts;
    opts.verbose = altro::Verbosity::Inner;
    opts.tol_cost = 1e-16;
    opts.tol_primal_feasibility = tol;
    opts.tol_stationarity = tol;
    opts.penalty_initial = mu_init;
    opts.iterations_max = max_iters;
    opts.use_backtracking_linesearch = true;
    solver.SetOptions(opts);
    ErrorCodes err = solver.Initialize();
    if (err != ErrorCodes::NoError) {
      throw std::runtime_error("Altro initialization failed");
    }

    // some testing

    SolveStatus status = solver.Solve();
    if (status != SolveStatus::Success) {
      fmt::println("Altro failed to solve problem.");
    } else {
      fmt::println("Altro successfully converged.");
    }

    std::vector<VectorXs> xs;
    for (int k = 0; k <= int(nsteps); k++) {
      xs.emplace_back(solver.GetStateDim(k));
      solver.GetState(xs.back().data(), k);
    }
    assert(xs.size() == nsteps + 1);

    matplot::figure();
    auto ax = matplot::gca();
    ax->title("Altro");
    ax->hold(true);
    for (long i = 0; i < 6; i++) {
      ax->plot(times, traj_coordinate(xs, i))->line_width(1.5);
    }
    ax->xlabel("Time");
    matplot::show();
  }

  return 0;
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
