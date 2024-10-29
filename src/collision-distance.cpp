#include "collision-distance.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace aligator_bench {

SphereCylinderCollisionDistance::SphereCylinderCollisionDistance(
    pin::Model model, int ndx, int nu, Vector2d center, double radius,
    std::vector<double> frame_radii, std::vector<pin::FrameIndex> frame_ids)
    : UnaryFunction(ndx, nu, int(frame_ids.size())), model_(model),
      cyl_center_(center), cyl_radius_(radius), frame_radii_(frame_radii),
      frame_ids_(frame_ids) {
  if (frame_radii_.size() != frame_ids_.size()) {
    ALIGATOR_RUNTIME_ERROR("Inconsistent number of frame AABB and IDs passed.");
  }
}

void SphereCylinderCollisionDistance::evaluate(const ConstVectorRef &x,
                                               StageFunctionData &data) const {
  Data &d = static_cast<Data &>(data);
  ConstVectorRef q = x.head(model_.nq);
  pin::Data &pdata = d.pin_data_;
  pin::forwardKinematics(model_, pdata, q);
  Vector2d err;

  using Eigen::Index;
  for (Index i = 0; i < Index(numCollisions()); i++) {
    pin::FrameIndex fid = frame_ids_[size_t(i)];
    auto &M = pin::updateFramePlacement(model_, pdata, fid);
    err = M.translation().head<2>() - cyl_center_;
    double margin = frame_radii_[size_t(i)];
    double res = err.squaredNorm() - std::pow(margin + cyl_radius_, 2.0);
    d.value_(Index(i)) = -res;
  }
}

void SphereCylinderCollisionDistance::computeJacobians(
    const ConstVectorRef &x, StageFunctionData &data) const {
  Data &d = static_cast<Data &>(data);
  pin::Data &pdata = d.pin_data_;
  const int nv = model_.nv;
  ConstVectorRef q = x.head(model_.nq);
  Vector2d err;

  using Eigen::Index;
  for (Index i = 0; i < Index(numCollisions()); i++) {
    pin::FrameIndex fid = frame_ids_[size_t(i)];
    pin::computeFrameJacobian(model_, pdata, q, fid, pin::LOCAL_WORLD_ALIGNED,
                              d.frame_jac_);
    auto &M = pdata.oMf[fid];
    err = M.translation().head<2>() - cyl_center_;
    d.Jx_.leftCols(nv).row(Index(i)).noalias() =
        -2.0 * err.transpose() * d.frame_jac_.topRows<2>();
  }
}

shared_ptr<StageFunctionData>
SphereCylinderCollisionDistance::createData() const {
  return std::make_shared<Data>(*this);
}

SphereCylinderCollisionDistance::Data::Data(
    const SphereCylinderCollisionDistance &model)
    : StageFunctionData(model), pin_data_(model.model_) {
  frame_jac_.resize(6, model.model_.nv);
}

} // namespace aligator_bench
