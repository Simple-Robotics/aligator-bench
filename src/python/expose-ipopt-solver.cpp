#include "common.h"
#include "ipopt-solver.hpp"

#include <aligator/core/traj-opt-problem.hpp>

#include <IpIpoptApplication.hpp>

void exposeIpoptSolver() {
  using namespace aligator_bench;
  bp::class_<SolverIpopt, boost::noncopyable>("SolverIpopt", bp::no_init)
      .def(bp::init<bool>(("self"_a, "rethrow_nonipopt_exceptions"_a = false)))
      .def("setup", &SolverIpopt::setup, ("self"_a, "problem"));
}
