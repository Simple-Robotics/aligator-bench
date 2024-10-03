#include "aligator-problem-to-altro.hpp"
#include "aligator-to-altro-types.hpp"

#include <aligator/core/traj-opt-problem.hpp>
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
    auto &stage = problem.stages_[size_t(k)];
    solver.SetDimension(stage->nx1(), stage->nu(), k, k);

    // cost
    auto [c, gc, Hc] = aligatorCostToAltro(stage->cost_);
    solver.SetCostFunction(c, gc, Hc, k, k);

    // dynamics
    auto [dyn, Jdyn] = aligatorExpDynamicsToAltro(stage->dynamics_);
    solver.SetExplicitDynamics(dyn, Jdyn, k, k);
  }

  {
    // terminal
    auto [tc, gtc, Htc] = aligatorCostToAltro(problem.term_cost_);
    solver.SetCostFunction(tc, gtc, Htc, horizon_length);
  }

  return psolver;
}

} // namespace aligator_bench
