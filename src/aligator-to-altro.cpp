#include "aligator-to-altro.hpp"
#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/explicit-dynamics.hpp>
#include <aligator/core/constraint.hpp>
#include <proxsuite-nlp/modelling/constraints.hpp>

#include <boost/unordered_map.hpp>

auto aligatorCostToAltro(xyz::polymorphic<CostAbstract> aliCost)
    -> altroCostTriplet {

  const auto data = aliCost->createData();
  using altro::a_float;
  altro::CostFunction f = [aliCost, data](const a_float *x_,
                                          const a_float *u_) -> a_float {
    const auto nx = aliCost->nx();
    const auto nu = aliCost->nu;
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, nu};
    aliCost->evaluate(x, u, *data);
    return data->value_;
  };
  altro::CostGradient gf = [aliCost, data](a_float *dx_, a_float *du_,
                                           const a_float *x_,
                                           const a_float *u_) {
    const auto nx = aliCost->nx();
    const auto nu = aliCost->nu;
    VecMap dx{dx_, nx};
    VecMap du{du_, nu};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, nu};
    aliCost->evaluate(x, u, *data);
    aliCost->computeGradients(x, u, *data);
    dx = data->Lx_;
    du = data->Lu_;
  };
  altro::CostHessian Hf = [aliCost, data](a_float *ddx_, a_float *ddu_,
                                          a_float *dxdu_, const a_float *x_,
                                          const a_float *u_) {
    const auto nx = aliCost->nx();
    const auto nu = aliCost->nu;
    MatMap ddx{ddx_, nx, nx};
    MatMap ddu{ddu_, nu, nu};
    MatMap dxdu{dxdu_, nx, nu};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, nu};
    aliCost->evaluate(x, u, *data);
    aliCost->computeGradients(x, u, *data);
    aliCost->computeHessians(x, u, *data);
    ddx = data->Lxx_;
    ddu = data->Luu_;
    dxdu = data->Lxu_;
  };
  return {f, gf, Hf};
}

auto aligatorExpDynamicsToAltro(xyz::polymorphic<ExplicitDynamics> dynamics)
    -> altroExplicitDynamics {
  auto data = std::static_pointer_cast<alcontext::ExplicitDynamicsData>(
      dynamics->createData());

  altro::ExplicitDynamicsFunction f = [dynamics,
                                       data](double *xnext_, const double *x_,
                                             const double *u_, float) {
    const auto nx1 = dynamics->nx1();
    const auto nu = dynamics->nu;
    const auto nx2 = dynamics->nx2();
    ConstVecMap x{x_, nx1};
    ConstVecMap u{u_, nu};
    VecMap xnext{xnext_, nx2};
    dynamics->forward(x, u, *data);
    xnext = data->xnext_;
  };
  altro::ExplicitDynamicsJacobian Jf =
      [dynamics, data](double *J_, const double *x_, const double *u_, float) {
        const auto nx1 = dynamics->nx1();
        const auto nu = dynamics->nu;
        const auto nx2 = dynamics->nx2();
        MatMap J{J_, nx2, nx1 + nu};
        ConstVecMap x{x_, nx1};
        ConstVecMap u{u_, nu};
        dynamics->forward(x, u, *data);
        dynamics->dForward(x, u, *data);
        J.leftCols(nx1) = data->Jx_;
        J.rightCols(nu) = data->Ju_;
      };
  return std::tuple{f, Jf};
}

using ZeroSet = proxsuite::nlp::EqualityConstraintTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;

boost::unordered_map<const std::type_info *, altro::ConstraintType>
    __aligatorConstraintRttiToAltro = {
        {&typeid(ZeroSet), altro::ConstraintType::EQUALITY},
        {&typeid(NegativeOrthant), altro::ConstraintType::INEQUALITY}};

// Return Rtti
altro::ConstraintType
aligatorConstraintAltroType(const alcontext::ConstraintSet &constraint) {
  return __aligatorConstraintRttiToAltro.at(&typeid(constraint));
}

altroConstraint
aligatorConstraintToAltro(int nx, alcontext::StageConstraint constraint) {
  using altro::a_float;
  auto data = constraint.func->createData();
  altro::ConstraintFunction c = [nx, f = constraint.func,
                                 data](a_float *val, const a_float *x_,
                                       const a_float *u_) {
    VecMap value{val, f->nr};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, f->nu};

    f->evaluate(x, u, x, *data);
    value = data->value_;
  };
  altro::ConstraintJacobian Jc = [nx, f = constraint.func,
                                  data](a_float *jac_, const a_float *x_,
                                        const a_float *u_) {
    VecMap jac{jac_, f->nr, f->ndx1 + f->nu};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, f->nu};
  };
  altro::ConstraintType ct = aligatorConstraintAltroType(*constraint.set);
  return {c, Jc, ct};
}
