#pragma once

#include "types.hpp"

#include <aligator/core/unary-function.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace aligator_bench {

namespace pin = pinocchio;
using aligator::shared_ptr;
using aligator::context::StageFunctionData;
using aligator::context::UnaryFunction;
using Eigen::Vector2d;

struct SphereCylinderCollisionDistance : UnaryFunction {
  pin::Model model_;
  Vector2d cyl_center_;
  double cyl_radius_;

  struct Data;

  SphereCylinderCollisionDistance(pin::Model model, int ndx, int nu,
                                  Vector2d center, double radius,
                                  std::vector<double> frame_radii,
                                  std::vector<pin::FrameIndex> frame_ids);

  void evaluate(const ConstVectorRef &x,
                StageFunctionData &data) const override;

  void computeJacobians(const ConstVectorRef &x,
                        StageFunctionData &data) const override;

  shared_ptr<StageFunctionData> createData() const override;

  size_t numCollisions() const { return frame_ids_.size(); }

private:
  std::vector<double> frame_radii_;
  std::vector<pin::FrameIndex> frame_ids_;
};

struct SphereCylinderCollisionDistance::Data : StageFunctionData {
  Data(const SphereCylinderCollisionDistance &model);

  pin::Data pin_data_;
  MatrixXs frame_jac_;
};

} // namespace aligator_bench
