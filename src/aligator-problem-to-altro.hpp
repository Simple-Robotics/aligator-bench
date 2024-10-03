#include <aligator/context.hpp>
#include <altro/altro.hpp>

namespace aligator_bench {
namespace alcontext = aligator::context;

altro::ALTROSolver
init_altro_from_aligator_problem(const alcontext::TrajOptProblem &problem);

} // namespace aligator_bench
