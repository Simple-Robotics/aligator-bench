import aligator
import pinocchio as pin
import example_robot_data as erd
import numpy as np
import matplotlib.pyplot as plt

from aligator import constraints, manifolds


class URProblem(object):
    robot = erd.load("ur5")
    EE_NAME = "tool0"

    def __init__(self, vel_constraint=False, ee_target=None):
        rmodel: pin.Model = self.robot.model.copy()

        if ee_target is None:
            ee_target = 0.98 * self.get_default_config_ee_pose(rmodel)

        self.problem, self.times = self._build_problem(
            rmodel, vel_constraint, ee_target, self.EE_NAME
        )
        self._vel_constraint = vel_constraint
        self.rmodel = rmodel

    @staticmethod
    def get_default_config_ee_pose(rmodel: pin.Model):
        q0 = pin.neutral(rmodel)
        rdata = rmodel.createData()
        ee_id = rmodel.getFrameId(URProblem.EE_NAME)
        pin.framesForwardKinematics(rmodel, rdata, q0)
        M_tool_q0: pin.SE3 = rdata.oMf[ee_id]
        return M_tool_q0.translation.copy()

    def name(self):
        return "UR5 Reach"

    @staticmethod
    def _build_problem(rmodel: pin.Model, vel_constraint, ee_target, ee_name: int):

        nv = rmodel.nv
        nu = nv

        space = manifolds.MultibodyPhaseSpace(rmodel)
        ndx = space.ndx

        x0 = space.neutral()
        xneut = space.neutral()
        dt = 0.01

        rcost = aligator.CostStack(space, nu)
        wx = 1e-4 * np.eye(ndx)
        rcost.addCost(aligator.QuadraticStateCost(space, nu, xneut, wx), dt)
        rcost.addCost(aligator.QuadraticControlCost(space, nu, 1e-3 * np.eye(nu)), dt)
        wx_term = 1e-8 * np.eye(ndx)
        term_cost = aligator.QuadraticStateCost(space, nu, xneut, wx_term)

        dyn = aligator.dynamics.IntegratorSemiImplEuler(
            aligator.dynamics.MultibodyFreeFwdDynamics(space), dt
        )

        tf = 1.0
        nsteps = int(tf / dt)
        times = np.linspace(0.0, tf, nsteps + 1)
        ee_id = rmodel.getFrameId(ee_name)

        def createTermConstraint(pf: np.ndarray):
            fn = aligator.FrameTranslationResidual(
                ndx, nu, rmodel, np.asarray(pf), ee_id
            )
            return fn, constraints.EqualityConstraintSet()

        problem = aligator.TrajOptProblem(x0, nu, space, term_cost)
        problem.addTerminalConstraint(*createTermConstraint(ee_target))

        for i in range(nsteps):
            stage = aligator.StageModel(rcost, dyn)
            if vel_constraint:
                vmax = rmodel.velocityLimit
                vmin = -vmax
                _b = np.concatenate([-vmax, vmin])
                _A = np.zeros((2 * nv, ndx))
                _A[:nv, nv:] = +np.eye(nv)
                _A[nv:, nv:] = -np.eye(nv)
                vel_box_fn = aligator.LinearFunction(ndx, nu, ndx, 2 * nv)
                vel_box_fn.A[:] = _A
                vel_box_fn.d[:] = _b
                stage.addConstraint(vel_box_fn, constraints.NegativeOrthant())
            problem.addStage(stage)
        return (problem, times)


if __name__ == "__main__":
    from .solver_runner import AltroRunner

    ur_problem = URProblem()
    problem = ur_problem.problem
    rmodel = ur_problem.rmodel
    nq = rmodel.nq
    times_ = ur_problem.times

    TOL = 1e-5
    mu_init = 1e-1

    MAX_ITER = 400
    alisolver = aligator.SolverProxDDP(TOL, mu_init, verbose=aligator.VERBOSE)
    alisolver.max_iters = MAX_ITER
    bcl_params: alisolver.AlmParams = alisolver.bcl_params
    bcl_params.mu_lower_bound = 1e-10
    alisolver.setup(problem)
    alisolver.run(problem)
    ali_iter = alisolver.results.num_iters

    print(alisolver.results)

    xs_ali = alisolver.results.xs.tolist()
    us_ali = alisolver.results.us.tolist()

    runner = AltroRunner({"verbose": True})
    altro_res = runner.solve(ur_problem, TOL)
    print(altro_res)
    altro_solve = runner.solver

    xs_altro = altro_solve.GetAllStates().tolist()
    us_altro = altro_solve.GetAllInputs().tolist()

    vs_ali = [x[nq:] for x in xs_ali]
    vs_altro = [x[nq:] for x in xs_altro]
    fig1, axes1 = aligator.utils.plotting.plot_controls_traj(times_, us_ali)
    fig2, axes2 = aligator.utils.plotting.plot_velocity_traj(times_, vs_ali, rmodel)
    _, axes3 = aligator.utils.plotting.plot_controls_traj(times_, us_altro, axes=axes1)
    _, axes4 = aligator.utils.plotting.plot_velocity_traj(
        times_, vs_altro, rmodel, axes=axes2
    )
    plt.show()
