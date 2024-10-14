#include "aligator-to-altro-types.hpp"
#include <aligator/core/cost-abstract.hpp>
#include <aligator/core/explicit-dynamics.hpp>
#include <aligator/core/constraint.hpp>
#include <proxsuite-nlp/modelling/constraints.hpp>

#include <boost/unordered_map.hpp>

namespace aligator_bench {

altroCostTriplet aligatorCostToAltro(xyz::polymorphic<CostAbstract> aliCost) {

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

altroExplicitDynamics
aligatorExpDynamicsToAltro(xyz::polymorphic<DynamicsModel> dynamics) {

#define _cast_to_explicit(d) dynamic_cast<const ExplicitDynamics &>(*d)

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
    _cast_to_explicit(dynamics).forward(x, u, *data);
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
        _cast_to_explicit(dynamics).forward(x, u, *data);
        _cast_to_explicit(dynamics).dForward(x, u, *data);
        J.leftCols(nx1) = data->Jx_;
        J.rightCols(nu) = data->Ju_;
      };
  return std::tuple{f, Jf};
#undef _cast_to_explicit
}

using ZeroSet = proxsuite::nlp::EqualityConstraintTpl<double>;
using NegativeOrthant = proxsuite::nlp::NegativeOrthantTpl<double>;
using BoxConstraint = proxsuite::nlp::BoxConstraintTpl<double>;

boost::unordered_map<std::type_index, altro::ConstraintType>
    __aligatorConstraintRttiToAltro = {
        {typeid(ZeroSet), altro::ConstraintType::EQUALITY},
        {typeid(NegativeOrthant), altro::ConstraintType::INEQUALITY}};

/// Reformulate as a two-sided inequality constraint
altroConstraint
_aligator_boxconstraint_handle(int nx, xyz::polymorphic<StageFunction> func,
                               const BoxConstraint &set) {
  using altro::a_float;
  auto data = func->createData();
  altro::ConstraintFunction c = [nx, f = func, set, data](a_float *val,
                                                          const a_float *x_,
                                                          const a_float *u_) {
    const int nr = f->nr;
    VecMap value{val, 2 * nr};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, f->nu};

    f->evaluate(x, u, *data);
    value.head(nr) = data->value_ - set.upper_limit;
    value.tail(nr) = set.lower_limit - data->value_;
  };
  altro::ConstraintJacobian Jc = [nx, f = func, data](a_float *jac_,
                                                      const a_float *x_,
                                                      const a_float *u_) {
    const int ndx = f->ndx1;
    const int nu = f->nu;
    const int nr = f->nr;
    MatMap jac{jac_, 2 * nr, ndx + nu};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, nu};
    f->evaluate(x, u, *data);
    f->computeJacobians(x, u, *data);
    jac.topLeftCorner(nr, ndx) = data->Jx_;
    jac.topRightCorner(nr, nu) = data->Ju_;
    jac.bottomRows(nr) = -jac.topRows(nr);
  };
  return {c, Jc, altro::ConstraintType::INEQUALITY, 2 * func->nr};
};

// Return Rtti
altro::ConstraintType
aligatorConstraintAltroType(const alcontext::ConstraintSet &constraint) {
  return __aligatorConstraintRttiToAltro.at(typeid(constraint));
}

altroConstraint aligatorConstraintToAltro(int nx,
                                          xyz::polymorphic<StageFunction> func,
                                          xyz::polymorphic<ConstraintSet> set) {
  if (auto ps = dynamic_cast<const BoxConstraint *>(&*set)) {
    return _aligator_boxconstraint_handle(nx, func, *ps);
  }

  using altro::a_float;
  auto data = func->createData();
  altro::ConstraintFunction c =
      [nx, f = func, data](a_float *val, const a_float *x_, const a_float *u_) {
        VecMap value{val, f->nr};
        ConstVecMap x{x_, nx};
        ConstVecMap u{u_, f->nu};

        f->evaluate(x, u, *data);
        value = data->value_;
      };
  altro::ConstraintJacobian Jc = [nx, f = func, data](a_float *jac_,
                                                      const a_float *x_,
                                                      const a_float *u_) {
    const int ndx = f->ndx1;
    const int nu = f->nu;
    MatMap jac{jac_, f->nr, ndx + nu};
    ConstVecMap x{x_, nx};
    ConstVecMap u{u_, nu};
    f->evaluate(x, u, *data);
    f->computeJacobians(x, u, *data);
    jac.leftCols(ndx) = data->Jx_;
    jac.rightCols(nu) = data->Ju_;
  };
  altro::ConstraintType ct = aligatorConstraintAltroType(*set);
  return {c, Jc, ct, func->nr};
}

} // namespace aligator_bench
