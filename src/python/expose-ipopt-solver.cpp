#include "common.h"
#include "ipopt-solver.hpp"

#include <aligator/core/traj-opt-problem.hpp>

#include <IpIpoptApplication.hpp>
#include <IpReturnCodes.hpp>

#define _get_ro(clsname, name) def_readonly(#name, &clsname::name##_)

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
      .def("setMaxIters", &SolverIpopt::setMaxiters, ("self"_a, "value"))
#define _c(name)                                                               \
  add_property(#name, bp::make_function(&SolverIpopt::name,                    \
                                        bp::return_internal_reference<>()))
      ._c(xs)
      ._c(us)
      ._c(lams)
      ._c(vs)
#undef _c
      ._get_ro(SolverIpopt, num_iter)
      ._get_ro(SolverIpopt, traj_cost)
      ._get_ro(SolverIpopt, dual_infeas)
      ._get_ro(SolverIpopt, cstr_violation)
      ._get_ro(SolverIpopt, complementarity);

  bp::enum_<Ipopt::ApplicationReturnStatus>("IpoptApplicationReturnStatus")
#define _c(name) value(#name, Ipopt::ApplicationReturnStatus::name)
      ._c(Solve_Succeeded)
      ._c(Solved_To_Acceptable_Level)
      ._c(Infeasible_Problem_Detected)
      ._c(Search_Direction_Becomes_Too_Small)
      ._c(Diverging_Iterates)
      ._c(User_Requested_Stop)
      ._c(Feasible_Point_Found)
      //
      ._c(Maximum_Iterations_Exceeded)
      ._c(Restoration_Failed)
      ._c(Error_In_Step_Computation)
      ._c(Maximum_CpuTime_Exceeded)
      ._c(Maximum_WallTime_Exceeded)
      //
      ._c(Not_Enough_Degrees_Of_Freedom)
      ._c(Invalid_Problem_Definition)
      ._c(Invalid_Option)
      ._c(Invalid_Number_Detected)
      //
      ._c(Unrecoverable_Exception)
      ._c(NonIpopt_Exception_Thrown)
      ._c(Insufficient_Memory)
      ._c(Internal_Error)
#undef _c
      ;
}
