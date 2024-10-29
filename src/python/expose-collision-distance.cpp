#include "common.h"
#include "collision-distance.hpp"

#include <aligator/python/polymorphic-convertible.hpp>

namespace pin = pinocchio;
using namespace aligator::context;

void exposeCollisionAvoidanceModel() {
#define _c(name) def_readwrite(#name, &SphereCylinderCollisionDistance::name)
  bp::class_<SphereCylinderCollisionDistance, bp::bases<UnaryFunction>>(
      "SphereCylinderCollisionDistance", bp::no_init)
      .def(
          bp::init<pin::Model, int, int, Vector2d, double, double,
                   pin::FrameIndex>(("self"_a, "model", "ndx", "nu", "center",
                                     "cyl_radius", "robot_radius", "frame_id")))
      ._c(model_)
      ._c(cyl_center_)
      ._c(cyl_radius_)
      ._c(robot_radius_)
      ._c(frame_id)
#undef _c
      .def(aligator::python::PolymorphicMultiBaseVisitor<StageFunction,
                                                         UnaryFunction>());
}
