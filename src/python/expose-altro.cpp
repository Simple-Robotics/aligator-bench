#include "common.h"

#include <altro/altro.hpp>

Eigen::VectorXd get_state_wrap(const altro::ALTROSolver &solver, int k) {
  Eigen::VectorXd x{solver.GetStateDim(k)};
  solver.GetState(x.data(), k);
  return x;
}

Eigen::VectorXd get_input_wrap(const altro::ALTROSolver &solver, int k) {
  Eigen::VectorXd u{solver.GetInputDim(k)};
  solver.GetInput(u.data(), k);
  return u;
}

auto altro_get_all_states(const altro::ALTROSolver &solver) {
  const size_t horizon_length = size_t(solver.GetHorizonLength());
  std::vector<Eigen::VectorXd> xs;
  for (size_t k = 0; k <= horizon_length; k++) {
    xs.push_back(get_state_wrap(solver, int(k)));
  }
  return xs;
}

auto altro_get_all_inputs(const altro::ALTROSolver &solver) {
  const size_t horizon_length = size_t(solver.GetHorizonLength());
  std::vector<Eigen::VectorXd> us;
  for (size_t k = 0; k < horizon_length; k++) {
    us.push_back(get_input_wrap(solver, int(k)));
  }
  return us;
}

void exposeAltro() {
  using namespace altro;

  // Expose a limited API to ALTRO, just enough to set solver options,
  // initialize, and solve.
  bp::class_<ALTROSolver, boost::noncopyable>(
      "ALTROSolver", "A wrapper class for the ALTRO solver", bp::no_init)
      // .def(bp::init<int>("self"_a)) // we should never init from Python
      .def(
          "GetOptions",
          +[](ALTROSolver &self) -> AltroOptions & {
            return self.GetOptions();
          },
          ("self"_a), bp::return_internal_reference<>())
      .def("SetOptions", &ALTROSolver::SetOptions, ("self"_a, "opts"))
      .def("CalcCost", &ALTROSolver::CalcCost, ("self"_a))
      .def("GetIterations", &ALTROSolver::GetIterations, ("self"_a))
      .def("GetPrimalFeasibility", &ALTROSolver::GetPrimalFeasibility,
           ("self"_a))
      .def("GetFinalObjective", &ALTROSolver::GetFinalObjective, ("self"_a))
      .def(
          "SetInitialState",
          +[](ALTROSolver &self, Eigen::Map<Eigen::VectorXd> x0) {
            self.SetInitialState(x0.data(), int(x0.size()));
          })
      .def("Initialize", &ALTROSolver::Initialize, ("self"_a))
      .def("Solve", &ALTROSolver::Solve, ("self"_a))
      .def("GetState", &get_state_wrap, ("self"_a, "k"))
      .def("GetInput", &get_input_wrap, ("self"_a, "k"))
      .def("GetAllStates", &altro_get_all_states, ("self"_a))
      .def("GetAllInputs", &altro_get_all_inputs, ("self"_a))
      .def("PrintStateTrajectory", &ALTROSolver::PrintStateTrajectory,
           ("self"_a))
      .def("PrintInputTrajectory", &ALTROSolver::PrintInputTrajectory,
           ("self"_a));

  bp::enum_<SolveStatus>("SolveStatus")
#define _c(name) value(#name, SolveStatus::name)
      ._c(Success)
      ._c(Unsolved)
      ._c(MaxIterations)
      ._c(MaxObjectiveExceeded)
      ._c(StateOutOfBounds)
      ._c(InputOutOfBounds)
      ._c(MeritFunGradientTooSmall);
#undef _c

  bp::enum_<Verbosity>("AltroVerbosity")
#define _c(name) value(#name, Verbosity::name)
      ._c(Silent)
      ._c(Outer)
      ._c(Inner)
      ._c(LineSearch);
#undef _c

  bp::enum_<ErrorCodes>("ErrorCodes")
#define _c(name) value(#name, ErrorCodes::name)
      ._c(NoError)
      ._c(StateDimUnknown)
      ._c(InputDimUnknown)
      ._c(NextStateDimUnknown)
      ._c(DimensionUnknown)
      ._c(BadIndex)
      ._c(DimensionMismatch)
      ._c(SolverNotInitialized)
      ._c(SolverAlreadyInitialized)
      ._c(NonPositive)
      ._c(TimestepNotPositive)
      ._c(CostFunNotSet)
      ._c(DynamicsFunNotSet)
      ._c(InvalidOptAtTerminalKnotPoint)
      ._c(MaxConstraintsExceeded)
      ._c(InvalidConstraintDim)
      ._c(CholeskyFailed)
      ._c(OpOnlyValidAtTerminalKnotPoint)
      ._c(InvalidPointer)
      ._c(BackwardPassFailed)
      ._c(LineSearchFailed)
      ._c(MeritFunctionGradientTooSmall)
      ._c(InvalidBoundConstraint)
      ._c(NonPositivePenalty)
      ._c(CostNotQuadratic)
      ._c(FileError);
#undef _c

  bp::class_<AltroOptions>("AltroOptions", bp::init<>("self"_a))
#define _c(name) def_readwrite(#name, &AltroOptions::name)
      ._c(iterations_max)
      ._c(tol_cost)
      ._c(tol_cost_intermediate)
      ._c(tol_primal_feasibility)
      ._c(tol_stationarity)
      ._c(tol_meritfun_gradient)
      ._c(max_state_value)
      ._c(max_input_value)
      ._c(penalty_initial)
      ._c(penalty_scaling)
      ._c(penalty_max)
      ._c(verbose)
      ._c(use_backtracking_linesearch);
#undef _c
}
