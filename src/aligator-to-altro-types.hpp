#pragma once

#include "./types.hpp"
#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>
#include <altro/solver/typedefs.hpp>

namespace aligator_bench {

namespace alcontext = aligator::context;
using alcontext::ConstraintSet;
using alcontext::CostAbstract;
using alcontext::DynamicsModel;
using alcontext::ExplicitDynamics;
using alcontext::StageFunction;

using altroCostTriplet =
    std::tuple<altro::CostFunction, altro::CostGradient, altro::CostHessian>;

using altroExplicitDynamics = std::tuple<altro::ExplicitDynamicsFunction,
                                         altro::ExplicitDynamicsJacobian>;

using altroConstraint =
    std::tuple<altro::ConstraintFunction, altro::ConstraintFunction,
               altro::ConstraintType, int>;

/// @brief Convert aligator cost function to altro
altroCostTriplet aligatorCostToAltro(xyz::polymorphic<CostAbstract> aliCost);

altroExplicitDynamics
aligatorExpDynamicsToAltro(xyz::polymorphic<DynamicsModel> dynamics);

altro::ConstraintType
aligatorConstraintAltroType(const ConstraintSet &constraint);

altroConstraint
aligatorConstraintToAltro(int nx, xyz::polymorphic<StageFunction> constraint,
                          xyz::polymorphic<ConstraintSet> set);

} // namespace aligator_bench
