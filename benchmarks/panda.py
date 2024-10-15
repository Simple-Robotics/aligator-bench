import aligator
import pinocchio as pin
import numpy as np
import example_robot_data as erd
import matplotlib.pyplot as plt
import time

from aligator import manifolds, constraints
from aligator.utils import plotting
from aligator_bench_pywrap import SolverIpopt
from pinocchio.visualize import MeshcatVisualizer
from tap import Tap
from typing import Literal


class Args(Tap):
    solver: Literal["ali", "ipopt", "altro"]
    viz: bool = False


args = Args().parse_args()
robot = erd.load("panda")
rmodel: pin.Model = robot.model
rmodel.gravity.np[:] = 0.0  # assume gravity is compensated
rdata = rmodel.createData()

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx
nv = rmodel.nv
nu = rmodel.nv

x0 = space.neutral()
q0 = rmodel.referenceConfigurations["default"]
x0[: rmodel.nq] = q0
term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, np.eye(ndx)))
ee_id = rmodel.getFrameId("panda_hand")
ee_ref0 = np.array([0.4, -0.5, 0.2])
ee_ref1 = np.array([0.1, 0.6, 0.6])
w_ee = np.eye(3) * 10.0
frame_res0 = aligator.FrameTranslationResidual(ndx, nu, rmodel, ee_ref0, ee_id)
frame_res1 = aligator.FrameTranslationResidual(ndx, nu, rmodel, ee_ref1, ee_id)
term_cost.addCost(aligator.QuadraticResidualCost(space, frame_res1, w_ee))
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, 1e-2 * np.eye(ndx)))

nsteps = 400
tf = 4.0
dt = tf / nsteps
times = np.linspace(0.0, tf, nsteps + 1)
print(f"tf: {tf} / dt: {dt}")

ode = aligator.dynamics.MultibodyFreeFwdDynamics(space)
dynamics = aligator.dynamics.IntegratorSemiImplEuler(ode, dt)


stages = []
for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)
    w_x = np.ones(ndx) * 1e-2
    w_x[nv:] = 1e-1
    rcost.addCost("xreg", aligator.QuadraticStateCost(space, nu, x0, dt * np.diag(w_x)))
    rcost.addCost(
        "ureg", aligator.QuadraticControlCost(space, nu, 1e-3 * dt * np.eye(nu))
    )
    if i == nsteps / 2:
        rcost.addCost("ee_pos", aligator.QuadraticResidualCost(space, frame_res0, w_ee))
    stage = aligator.StageModel(rcost, dynamics)
    effLim = rmodel.effortLimit
    stage.addConstraint(
        aligator.ControlErrorResidual(ndx, np.zeros(nu)),
        constraints.BoxConstraint(-effLim, +effLim),
    )

    stages.append(stage)

problem = aligator.TrajOptProblem(x0, stages, term_cost)
problem.addTerminalConstraint(frame_res1, constraints.EqualityConstraintSet())

TOL = 1e-4
mu_init = 1e-4
max_iter = 400

match args.solver:
    case "ali":
        solver = aligator.SolverProxDDP(TOL, mu_init)
        solver.rollout_type = aligator.ROLLOUT_LINEAR
        solver.max_iters = max_iter
        solver.setup(problem)
        _start = time.time_ns()
        ret = solver.run(problem)
        _elapsed = time.time_ns() - _start
        print(f"Elapsed: {_elapsed * 1e-6} ms")
        print(solver.results)
        xs_sol = solver.results.xs.tolist()
        us_sol = solver.results.us.tolist()
    case "ipopt":
        solver = SolverIpopt()
        solver.setup(problem, True)
        solver.setOption("tol", TOL)
        solver.setPrintLevel(5)
        solver.setMaxIters(max_iter)
        _start = time.time_ns()
        ret = solver.solve()
        _elapsed = time.time_ns() - _start
        print(f"Elapsed: {_elapsed * 1e-6} ms")
        xs_sol = solver.xs.tolist()
        us_sol = solver.us.tolist()
    case "altro":
        from aligator_bench_pywrap import (
            ALTROSolver,
            AltroVerbosity,
            initAltroFromAligatorProblem,
        )

        _altrosolver: ALTROSolver = initAltroFromAligatorProblem(problem)
        init_code = _altrosolver.Initialize()
        _altro_opts = _altrosolver.GetOptions()
        _altro_opts.iterations_max = max_iter
        _altro_opts.tol_cost = 1e-16
        _altro_opts.tol_primal_feasibility = TOL
        _altro_opts.tol_stationarity = TOL
        _altro_opts.verbose = AltroVerbosity.Inner
        _altro_opts.use_backtracking_linesearch = True
        print(init_code)
        _start = time.time_ns()
        ret = _altrosolver.Solve()
        _elapsed = time.time_ns() - _start
        print(f"Elapsed: {_elapsed * 1e-6} ms")
        print(ret)
        xs_sol = _altrosolver.GetAllStates().tolist()
        us_sol = _altrosolver.GetAllInputs().tolist()
    case _:
        raise ValueError("Derp")


def get_ee_traj(xs, rmodel: pin.Model):
    rdata = rmodel.createData()

    ee_traj = []
    for i in range(nsteps):
        q = xs_sol[i][: rmodel.nq]
        pin.framesForwardKinematics(rmodel, rdata, q)
        ee_traj.append(rdata.oMf[ee_id].translation.copy())
    ee_traj = np.asarray(ee_traj)
    return ee_traj


plotting.plot_controls_traj(times, us_sol, effort_limit=effLim, rmodel=rmodel)
plt.show()

if args.viz:
    viz = MeshcatVisualizer(
        rmodel, visual_model=robot.visual_model, visual_data=robot.visual_data
    )
    viz.initViewer(open=True, loadModel=True)
    viz.display(q0)

    input("Press [enter]")
    qs = [x[: rmodel.nq] for x in xs_sol]
    viz.play(q_trajectory=qs, dt=dt)
