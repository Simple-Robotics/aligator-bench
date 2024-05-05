#pragma once

#include "solvers.hpp"
#include <aligator/core/results-base.hpp>
#include <aligator/core/solver-util.hpp>

namespace aligator_bench {

/// @brief "Naive" augmented Lagrangian DDP baseline.
class AugLagDdp {
  using Results = aligator::ResultsBaseTpl<Scalar>;
  using Workspace = aligator::WorkspaceBaseTpl<Scalar>;

  AugLagDdp();

  Results results_;
};

} // namespace aligator_bench
