import aligator
import pinocchio as pin
import example_robot_data as erd
import numpy as np
import matplotlib.pyplot as plt

from aligator import constraints, manifolds


class URProblem(object):
    EE_NAME = "tool0"

    def __init__(self, random_seed: int, vel_constraint=False):
        robot = erd.load("ur5")
        rmodel: pin.Model = robot.model
        q0 = pin.neutral(rmodel)
        rdata = rmodel.createData()
        _ee_id = rmodel.getFrameId(self.EE_NAME)
        pin.framesForwardKinematics(rmodel, rdata, q0)

        # todo: change
        M_tool_q0: pin.SE3 = rdata.oMf[_ee_id]
        ee_target = 0.9 * M_tool_q0.translation.copy()

        self.problem, self.times = self._build_problem(
            rmodel, random_seed, vel_constraint, ee_target, self.EE_NAME
        )
        self.rmodel = rmodel

    @staticmethod
    def _build_problem(
        rmodel: pin.Model, random_seed, vel_constraint, ee_target, ee_name: int
    ):
        pin.seed(random_seed)
        np.random.seed(random_seed)

        nv = rmodel.nv
        nu = nv

        space = manifolds.MultibodyPhaseSpace(rmodel)
        ndx = space.ndx

        x0 = space.neutral()
        dt = 0.01

        rcost = aligator.CostStack(space, nu)
        wx = 1e-4 * np.eye(ndx)
        rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, wx), dt)
        rcost.addCost(aligator.QuadraticControlCost(space, nu, 1e-3 * np.eye(nu)), dt)
        wx_term = 1e-8 * np.eye(ndx)
        term_cost = aligator.QuadraticStateCost(space, nu, x0, wx_term)

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
            return aligator.StageConstraint(fn, constraints.EqualityConstraintSet())

        problem = aligator.TrajOptProblem(x0, nu, space, term_cost)
        problem.addTerminalConstraint(createTermConstraint(ee_target))

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
    import aligator_bench_pywrap
    from .solver_runner import AltroRunner

    ur_problem = URProblem(42)
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

    runner.solve(ur_problem, TOL)

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
