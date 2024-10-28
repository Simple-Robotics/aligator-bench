#include "collision-distance.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

SphereCylinderCollisionDistance::SphereCylinderCollisionDistance(
    pin::Model model, int ndx, int nu, Vector2d center, double radius,
    double robot_radius, pin::FrameIndex frame_id)
    : UnaryFunction(ndx, nu, 1), model_(model), cyl_center_(center),
      cyl_radius_(radius), robot_radius_(robot_radius), frame_id(frame_id) {}

void SphereCylinderCollisionDistance::evaluate(const ConstVectorRef &x,
                                               StageFunctionData &data) const {
  Data &d = static_cast<Data &>(data);
  ConstVectorRef q = x.head(model_.nq);
  pin::forwardKinematics(model_, d.pin_data_, q);
  auto &M = pin::updateFramePlacement(model_, d.pin_data_, frame_id);
  Vector2d err = M.translation().head<2>() - cyl_center_;
  double res = err.squaredNorm() - std::pow(cyl_radius_, 2.0);
  d.value_(0) = res;
}

void SphereCylinderCollisionDistance::computeJacobians(
    const ConstVectorRef &x, StageFunctionData &data) const {
  Data &d = static_cast<Data &>(data);
  int nv = model_.nv;
  ConstVectorRef q = x.head(model_.nq);
  pin::computeFrameJacobian(model_, d.pin_data_, q, frame_id,
                            pin::LOCAL_WORLD_ALIGNED, d.frame_jac_);
  auto &M = d.pin_data_.oMf[frame_id];
  Vector2d err = M.translation().head<2>() - cyl_center_;
  d.Jx_.leftCols(nv).noalias() =
      -2.0 * d.frame_jac_.leftCols(2).transpose() * err;
}

shared_ptr<StageFunctionData>
SphereCylinderCollisionDistance::createData() const {
  return std::make_shared<Data>(*this);
}

SphereCylinderCollisionDistance::Data::Data(
    const SphereCylinderCollisionDistance &model)
    : StageFunctionData(model) {
  frame_jac_.resize(6, model.model_.nv);
}
