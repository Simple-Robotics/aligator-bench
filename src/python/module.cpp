#include <eigenpy/eigenpy.hpp>
#include <altro/altro.hpp>
#include <aligator/python/utils.hpp>

namespace bp = boost::python;

bp::arg operator""_a(const char *name, std::size_t) { return bp::arg(name); }

void exposeAltro() {
  using namespace altro;
  bp::scope scope = aligator::python::get_namespace("altro_wrap");

  // Expose a limited API to ALTRO, just enough to set solver options,
  // initialize, and solve.
  bp::class_<ALTROSolver, boost::noncopyable>(
      "ALTROSolver", "A wrapper class for the ALTRO solver", bp::no_init)
      // .def(bp::init<int>("self"_a)) // we should never init from Python
      .def("SetOptions", &ALTROSolver::SetOptions, ("self"_a, "opts"))
      .def(
          "SetInitialState",
          +[](ALTROSolver &self, Eigen::Map<Eigen::VectorXd> x0) {
            self.SetInitialState(x0.data(), int(x0.size()));
          })
      .def("Initialize", &ALTROSolver::Initialize, ("self"_a))
      .def("Solve", &ALTROSolver::Solve, ("self"_a))
      .def(
          "GetState",
          +[](const ALTROSolver &self, int k) {
            Eigen::VectorXd x{self.GetStateDim(k)};
            self.GetState(x.data(), k);
            return x;
          })
      .def(
          "GetInput",
          +[](const ALTROSolver &self, int k) {
            Eigen::VectorXd u{self.GetInputDim(k)};
            self.GetInput(u.data(), k);
            return u;
          })
      .def("PrintStateTrajectory", &ALTROSolver::PrintStateTrajectory,
           ("self"_a))
      .def("PrintInputTrajectory", &ALTROSolver::PrintInputTrajectory,
           ("self"_a));

  bp::enum_<Verbosity>("Verbosity")
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
      ._c(tol_stationarity)
      ._c(tol_meritfun_gradient)
      ._c(max_state_value)
      ._c(max_input_value)
      ._c(penalty_initial)
      ._c(penalty_scaling)
      ._c(penalty_max);
#undef _c
}

BOOST_PYTHON_MODULE(aligator_bench_pywrap) { exposeAltro(); }
