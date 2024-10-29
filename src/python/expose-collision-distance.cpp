#include "common.h"
#include "collision-distance.hpp"

#include <aligator/python/polymorphic-convertible.hpp>

namespace pin = pinocchio;
using namespace aligator::context;
using aligator_bench::SphereCylinderCollisionDistance;

void exposeCollisionAvoidanceModel() {
#define _c(name) def_readwrite(#name, &SphereCylinderCollisionDistance::name)
  bp::class_<SphereCylinderCollisionDistance, bp::bases<UnaryFunction>>(
      "SphereCylinderCollisionDistance", bp::no_init)
      .def(bp::init<pin::Model, int, int, Eigen::Vector2d, double,
                    std::vector<double>, std::vector<pin::FrameIndex>>(
          ("self"_a, "model", "ndx", "nu", "center", "cyl_radius",
           "frame_radii", "frame_ids")))
      ._c(model_)
      ._c(cyl_center_)
      ._c(cyl_radius_)
#undef _c
      .add_property("numCollisions",
                    &SphereCylinderCollisionDistance::numCollisions)
      .def(aligator::python::PolymorphicMultiBaseVisitor<StageFunction,
                                                         UnaryFunction>());
}
