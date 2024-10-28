#include <aligator/context.hpp>
#include <aligator/core/unary-function.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

namespace pin = pinocchio;
using namespace aligator::context;
using aligator::shared_ptr;
using Eigen::Vector2d;

struct SphereCylinderCollisionDistance : UnaryFunction {
  pin::Model model_;
  Vector2d cyl_center_;
  double cyl_radius_;
  double robot_radius_;
  pin::FrameIndex frame_id;

  struct Data;

  SphereCylinderCollisionDistance(pin::Model model, int ndx, int nu,
                                  Vector2d center, double radius,
                                  double robot_radius,
                                  pin::FrameIndex frame_id);

  void evaluate(const ConstVectorRef &x,
                StageFunctionData &data) const override;

  void computeJacobians(const ConstVectorRef &x,
                        StageFunctionData &data) const override;

  shared_ptr<StageFunctionData> createData() const override;
};

struct SphereCylinderCollisionDistance::Data : StageFunctionData {
  Data(const SphereCylinderCollisionDistance &model);

  pin::Data pin_data_;
  MatrixXs frame_jac_;
};
