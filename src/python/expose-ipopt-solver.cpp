#include "common.h"
#include "ipopt-solver.hpp"

#include <aligator/core/traj-opt-problem.hpp>

#include <IpIpoptApplication.hpp>

void exposeIpoptSolver() {
  using namespace aligator_bench;
#define _set_opt(type)                                                         \
  def<void (SolverIpopt::*)(const std::string &name, type value)>(             \
      "setOption", &SolverIpopt::setOption, ("self"_a, "name", "value"))
  bp::class_<SolverIpopt, boost::noncopyable>("SolverIpopt", bp::no_init)
      .def(bp::init<bool>(("self"_a, "rethrow_nonipopt_exceptions"_a = false)))
      .def("setup", &SolverIpopt::setup,
           ("self"_a, "problem", "verbose"_a = false))
      .def("solve", &SolverIpopt::solve, ("self"_a))
      ._set_opt(const std::string &)
      ._set_opt(int)
      ._set_opt(double)
#undef _set_opt
#define _c(name)                                                               \
  add_property(#name, bp::make_function(&SolverIpopt::name,                    \
                                        bp::return_internal_reference<>()))
      ._c(xs)
      ._c(us)
      ._c(lams)
      ._c(vs)
#undef _c
      ;
}
