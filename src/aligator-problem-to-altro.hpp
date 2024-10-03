#include <aligator/context.hpp>
#include <altro/altro.hpp>

namespace aligator_bench {
namespace alcontext = aligator::context;

altro::ALTROSolver *
initAltroFromAligatorProblem(const alcontext::TrajOptProblem &problem);

} // namespace aligator_bench
