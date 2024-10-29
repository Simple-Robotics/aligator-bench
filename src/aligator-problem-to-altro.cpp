#include "aligator-problem-to-altro.hpp"
#include "aligator-to-altro-types.hpp"

#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/explicit-dynamics.hpp>

namespace aligator_bench {
altro::ALTROSolver *
initAltroFromAligatorProblem(const alcontext::TrajOptProblem &problem) {
  const auto horizon_length = (int)problem.numSteps();
  const auto x0 = problem.getInitState();
  auto *psolver = new altro::ALTROSolver{horizon_length};
  altro::ALTROSolver &solver = *psolver;
  solver.SetInitialState(x0.data(), int(x0.size()));

  auto constraint_handle = [&solver](int nx, int k,
                                     const alcontext::ConstraintStack &cstrs) {
    // constraints
    for (size_t j = 0; j < cstrs.size(); k++) {
      auto [c, Jc, ct, dim] =
          aligatorConstraintToAltro(nx, cstrs.funcs[j], cstrs.sets[j]);
      std::string label = fmt::format("cstr_{:d}", j);
      solver.SetConstraint(c, Jc, dim, ct, label, k, k + 1);
    }
  };

  solver.SetTimeStep(1.0); // Aligator already handles the timestep
  for (int k = 0; k < horizon_length; k++) {
    const auto &stage = problem.stages_[size_t(k)];
    const int nx = stage->nx1();
    solver.SetDimension(stage->nx1(), stage->nu(), k, k + 1);

    // cost
    auto [c, gc, Hc] = aligatorCostToAltro(stage->cost_);
    solver.SetCostFunction(c, gc, Hc, k, k + 1);

    // dynamics
    auto [dyn, Jdyn] = aligatorExpDynamicsToAltro(stage->dynamics_);
    solver.SetExplicitDynamics(dyn, Jdyn, k, k + 1);

    // constraints
    constraint_handle(nx, k, stage->constraints_);
  }

  {
    // terminal
    const int nxterm = problem.term_cost_->nx();
    solver.SetDimension(nxterm, problem.term_cost_->nu, horizon_length);
    auto [tc, gtc, Htc] = aligatorCostToAltro(problem.term_cost_);
    solver.SetCostFunction(tc, gtc, Htc, horizon_length);
    constraint_handle(nxterm, horizon_length, problem.term_cstrs_);
  }

  return psolver;
}

} // namespace aligator_bench
