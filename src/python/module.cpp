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
            Eigen::VectorXd out{self.GetStateDim(k)};
            self.GetState(out.data(), k);
            return out;
          })
      .def(
          "GetInput", +[](const ALTROSolver &self, int k) {
            Eigen::VectorXd out{self.GetInputDim(k)};
            self.GetInput(out.data(), k);
            return out;
          });

  bp::enum_<Verbosity>("AltroVerbosity")
      .value("Silent", Verbosity::Silent)
      .value("Outer", Verbosity::Outer)
      .value("Inner", Verbosity::Inner)
      .value("LineSearch", Verbosity::LineSearch);

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
