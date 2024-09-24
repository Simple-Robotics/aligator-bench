#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/modelling/costs/quad-costs.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <aligator/modelling/dynamics/multibody-free-fwd.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

#include "aligator-to-altro.hpp"
#include "robots/robot_load.hpp"

#include <altro/altro.hpp>
#include <matplot/matplot.h>

namespace pin = pinocchio;
using alcontext::MatrixXs;
using alcontext::StageModel;
using alcontext::TrajOptProblem;
using alcontext::VectorXs;

using MultibodyFreeFwd =
    aligator::dynamics::MultibodyFreeFwdDynamicsTpl<double>;
using Space = proxsuite::nlp::MultibodyPhaseSpace<double>;
using aligator::dynamics::IntegratorSemiImplEulerTpl;

auto createCost(Space space, double dt) {
  const pin::Model &model = space.getModel();
  const auto nu = model.nv;
  const auto ndx = space.ndx();
  aligator::CostStackTpl<double> costs{space, nu};

  MatrixXs Wx{ndx, ndx};
  Wx.setIdentity();

  MatrixXs Wu{nu, nu};
  Wu.setIdentity();

  costs.addCost("quad", aligator::QuadraticCostTpl<double>{Wx, Wu}, 1e-2 * dt);
  return costs;
}

auto createTerminalCost(Space space, Eigen::Vector3d xf) {

  const auto fid = space.getModel().getFrameId("tool0");
  const auto &model = space.getModel();
  const auto nu = model.nv;
  aligator::FrameTranslationResidualTpl<double> res{space.ndx(), nu, model, xf,
                                                    fid};
  Eigen::Matrix3d wr;
  wr.setIdentity();
  return aligator::QuadraticResidualCostTpl<double>{space, std::move(res), wr};
}

auto traj_coordinate(std::span<VectorXs> states, long i) {
  std::vector<double> out;
  for (const auto &x : states) {
    assert(i < x.size());
    out.push_back(x[i]);
  }
  return out;
}

int main() {
  const double dt = 5e-2;
  const size_t nsteps = 10;
  const double tf = nsteps * dt;
  const std::vector<double> times = matplot::linspace(0, tf, nsteps + 1);

  pin::Model model = aligator_bench::loadModelFromToml("ur.toml", "ur5");
  std::cout << model << std::endl;

  const Space state_space{model};
  VectorXs x0 = state_space.neutral();
  x0.head<3>() << 0.1, -0.64, 1.38;

  const IntegratorSemiImplEulerTpl<double> dynamics{
      MultibodyFreeFwd{state_space}, dt};
  const auto rcost = createCost(state_space, dt);

  Eigen::Vector3d xf{0.5, 0.5, 0.1};
  const auto term_cost = createTerminalCost(state_space, xf);

  StageModel stage{rcost, dynamics};

  std::vector<xyz::polymorphic<StageModel>> stages{nsteps, stage};
  TrajOptProblem problem{x0, stages, term_cost};

  const double mu_init = 1e-10;
  const double tol = 1e-3;
  alcontext::SolverProxDDP solver{tol, mu_init};
  solver.verbose_ = aligator::VERBOSE;
  solver.linear_solver_choice = aligator::LQSolverChoice::SERIAL;

  solver.setup(problem);
  solver.run(problem);
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
    // create ALTRO problem
    altro::ALTROSolver solver{nsteps};
    altro::AltroOptions opts;
    opts.verbose = altro::Verbosity::Inner;
    opts.tol_cost = 1e-16;
    opts.tol_primal_feasibility = tol;
    opts.tol_stationarity = tol;
    opts.use_backtracking_linesearch = true;
    solver.SetOptions(opts);
    const int nx = stage.nx1();
    const int nu = stage.nu();
    auto [c, gc, Hc] = aligatorCostToAltro(rcost);
    auto [tc, gtc, Htc] = aligatorCostToAltro(term_cost);
    auto [dyn, Jdyn] = aligatorExpDynamicsToAltro(dynamics);

    solver.SetDimension(nx, nu);
    solver.SetTimeStep(1.0);
    solver.SetCostFunction(c, gc, Hc);
    solver.SetCostFunction(tc, gtc, Htc, 10);
    solver.SetExplicitDynamics(dyn, Jdyn);
    solver.SetInitialState(x0.data(), nx);
    ErrorCodes err = solver.Initialize();
    if (err != ErrorCodes::NoError) {
      throw std::runtime_error("Altro initialization failed");
    }

    // some testing

    SolveStatus status = solver.Solve();
    if (status != SolveStatus::Success) {
      throw std::runtime_error("Altro failed to solve problem.");
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
