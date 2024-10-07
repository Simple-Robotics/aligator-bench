#pragma once
#include <aligator/context.hpp>

namespace aligator_bench {
using aligator::context::MatrixXs;
using aligator::context::VectorXs;

using ConstVecMap = Eigen::Map<const VectorXs>;
using VecMap = Eigen::Map<VectorXs>;
using ConstMatMap = Eigen::Map<const MatrixXs>;
using MatMap = Eigen::Map<MatrixXs>;

} // namespace aligator_bench
