#include "common.h"
#include <aligator/core/traj-opt-problem.hpp>
#include "aligator-problem-to-altro.hpp"

using namespace aligator_bench;

void exposeAligatorToAltro() {
  bp::import("aligator");
  bp::def("initAltroFromAligatorProblem", initAltroFromAligatorProblem,
          ("aligator_problem"_a),
          "Instantiate an Altro solver from an Aligator problem.",
          bp::return_value_policy<bp::manage_new_object>());
}
