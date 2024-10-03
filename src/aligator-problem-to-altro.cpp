#include "aligator-problem-to-altro.hpp"
#include "aligator-to-altro-types.hpp"

#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/explicit-dynamics.hpp>

namespace aligator_bench {
altro::ALTROSolver *
initAltroFromAligatorProblem(const alcontext::TrajOptProblem &problem) {
  auto horizon_length = (int)problem.numSteps();
  auto x0 = problem.getInitState();
  auto *psolver = new altro::ALTROSolver{horizon_length};
  altro::ALTROSolver &solver = *psolver;
  solver.SetInitialState(x0.data(), int(x0.size()));

  solver.SetTimeStep(1.0); // Aligator already handles the timestep
  for (int k = 0; k < horizon_length; k++) {
    fmt::println("Adding stage {:d}", k);
    auto &stage = problem.stages_[size_t(k)];
    solver.SetDimension(stage->nx1(), stage->nu(), k, k + 1);
    fmt::println("> Set dimension");

    // cost
    auto [c, gc, Hc] = aligatorCostToAltro(stage->cost_);
    solver.SetCostFunction(c, gc, Hc, k, k + 1);
    fmt::println("> Set cost");

    // dynamics
    auto [dyn, Jdyn] = aligatorExpDynamicsToAltro(stage->dynamics_);
    solver.SetExplicitDynamics(dyn, Jdyn, k, k + 1);
    fmt::println("> Set dynamics");
  }

  {
    // terminal
    int nxterm = problem.term_cost_->nx();
    solver.SetDimension(nxterm, problem.term_cost_->nu, horizon_length);
    auto [tc, gtc, Htc] = aligatorCostToAltro(problem.term_cost_);
    solver.SetCostFunction(tc, gtc, Htc, horizon_length);
    fmt::println("> Set terminal cost");
  }

  return psolver;
}

} // namespace aligator_bench
