#include <aligator/core/constraint.hpp>
#include <aligator/modelling/multibody/frame-translation.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

using Space = proxsuite::nlp::MultibodyPhaseSpace<double>;
namespace alcontext = aligator::context;

xyz::polymorphic<alcontext::StageFunction>
createUr5EeResidual(Space space, Eigen::Vector3d ee_pos) {
  using aligator::FrameTranslationResidualTpl;
  const auto frame_id = space.getModel().getFrameId("tool0");
  const auto &model = space.getModel();
  const auto ndx = space.ndx();
  const auto nu = model.nv;
  return FrameTranslationResidualTpl<double>{ndx, nu, model, ee_pos, frame_id};
}
