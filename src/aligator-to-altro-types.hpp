#pragma once

#include <aligator/context.hpp>
#include <proxsuite-nlp/third-party/polymorphic_cxx14.hpp>
#include <altro/solver/typedefs.hpp>

namespace alcontext = aligator::context;
using alcontext::ConstraintSet;
using alcontext::CostAbstract;
using alcontext::DynamicsModel;
using alcontext::ExplicitDynamics;
using alcontext::StageFunction;

/* Typedefs */

using ConstVecMap = Eigen::Map<const alcontext::VectorXs>;
using VecMap = Eigen::Map<alcontext::VectorXs>;
using ConstMatMap = Eigen::Map<const alcontext::MatrixXs>;
using MatMap = Eigen::Map<alcontext::MatrixXs>;

using altroCostTriplet =
    std::tuple<altro::CostFunction, altro::CostGradient, altro::CostHessian>;

using altroExplicitDynamics = std::tuple<altro::ExplicitDynamicsFunction,
                                         altro::ExplicitDynamicsJacobian>;

using altroConstraint =
    std::tuple<altro::ConstraintFunction, altro::ConstraintFunction,
               altro::ConstraintType>;

/// @brief Convert aligator cost function to altro
altroCostTriplet aligatorCostToAltro(xyz::polymorphic<CostAbstract> aliCost);

altroExplicitDynamics
aligatorExpDynamicsToAltro(xyz::polymorphic<DynamicsModel> dynamics);

altro::ConstraintType
aligatorConstraintAltroType(const alcontext::ConstraintSet &constraint);

altroConstraint
aligatorConstraintToAltro(int nx, xyz::polymorphic<StageFunction> constraint,
                          xyz::polymorphic<ConstraintSet> set);
