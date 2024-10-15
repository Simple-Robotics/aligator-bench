import aligator
import pinocchio as pin
import example_robot_data as erd
import numpy as np
import matplotlib.pyplot as plt

from aligator import constraints, manifolds


def get_default_config_ee_pose(rmodel: pin.Model, ee_name: str):
    q0 = pin.neutral(rmodel)
    rdata = rmodel.createData()
    ee_id = rmodel.getFrameId(ee_name)
    pin.framesForwardKinematics(rmodel, rdata, q0)
    M_tool_q0: pin.SE3 = rdata.oMf[ee_id]
    return M_tool_q0.translation.copy()


class URProblem(object):
    robot = erd.load("ur5")
    EE_NAME = "tool0"

    def __init__(self, vel_constraint=False, ee_target=None):
        rmodel: pin.Model = self.robot.model.copy()
        q0 = pin.neutral(rmodel)

        if ee_target is None:
            ee_target = 0.7 * get_default_config_ee_pose(rmodel, self.EE_NAME)

        self.problem, self.times = self._build_problem(
            rmodel, q0, vel_constraint, ee_target, self.EE_NAME
        )
        self._vel_constraint = vel_constraint
        self.rmodel = rmodel

    def name(self):
        return "UR5 Reach"

    @staticmethod
    def _build_problem(rmodel: pin.Model, q0, vel_constraint, ee_target, ee_name: int):
        nv = rmodel.nv
        nu = nv

        space = manifolds.MultibodyPhaseSpace(rmodel)
        ndx = space.ndx

        x0 = np.concatenate((q0, np.zeros(nv)))
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
                stage.addConstraint(
                    aligator.StateErrorResidual(space, nu, xneut)[nv:],
                    constraints.BoxConstraint(vmin, vmax),
                )
            problem.addStage(stage)
        return (problem, times)


if __name__ == "__main__":
    from tap import Tap
    from .solver_runner import ProxDdpRunner, AltroRunner

    class Args(Tap):
        plot: bool = False

    args = Args().parse_args()
    ur_problem = URProblem(True)
    rmodel = ur_problem.rmodel
    nq = rmodel.nq
    times_ = ur_problem.times

    TOL = 1e-3
    mu_init = 1e1

    MAX_ITER = 400
    alirunner = ProxDdpRunner(
        {
            "rollout_type": aligator.ROLLOUT_LINEAR,
            "verbose": True,
            "max_iters": MAX_ITER,
            "mu_init": mu_init,
        }
    )
    alirunner.solve(ur_problem, TOL)
    alisolver = alirunner.solver

    ali_iter = alisolver.results.num_iters

    print(alisolver.results)

    xs_ali = alisolver.results.xs.tolist()
    us_ali = alisolver.results.us.tolist()
    vs_ali = [x[nq:] for x in xs_ali]

    altrorunner = AltroRunner(
        {
            "verbose": True,
            "max_iters": MAX_ITER,
            "mu_init": mu_init,
            "tol_stationarity": mu_init,
        }
    )
    altro_res = altrorunner.solve(ur_problem, TOL)
    print(altro_res)
    altro_solve = altrorunner.solver

    xs_altro = altro_solve.GetAllStates().tolist()
    us_altro = altro_solve.GetAllInputs().tolist()
    vs_altro = [x[nq:] for x in xs_altro]

    # iprunner = IpoptRunner({"max_iters": MAX_ITER})
    # ipres = iprunner.solve(ur_problem, TOL)
    # print("IpoptRunner result:", ipres)
    # ipsolve = iprunner.solver

    # xs_ipo = ipsolve.xs.tolist()
    # us_ipo = ipsolve.us.tolist()
    # vs_ipo = [x[nq:] for x in xs_ipo]

    if args.plot:
        fig1, axes1 = aligator.utils.plotting.plot_controls_traj(times_, us_ali)
        fig2, axes2 = aligator.utils.plotting.plot_velocity_traj(times_, vs_ali, rmodel)
        _, axes3 = aligator.utils.plotting.plot_controls_traj(
            times_, us_altro, axes=axes1
        )
        _, axes4 = aligator.utils.plotting.plot_velocity_traj(
            times_, vs_altro, rmodel, axes=axes2
        )
        plt.show()
