#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/dynamics/integrator-semi-euler.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

namespace alcontext = aligator::context;
using Space = proxsuite::nlp::MultibodyPhaseSpace<double>;

aligator::CostStackTpl<double> createUr5Cost(Space space, double dt);

xyz::polymorphic<alcontext::StageFunction>
createUr5EeResidual(Space space, Eigen::Vector3d ee_pos);

aligator::dynamics::IntegratorSemiImplEulerTpl<double>
createDynamics(Space space, double dt);

xyz::polymorphic<alcontext::CostAbstract>
createTerminalCost(Space space, Eigen::Vector3d ee_pos);
xyz::polymorphic<alcontext::CostAbstract> createRegTerminalCost(Space space);
