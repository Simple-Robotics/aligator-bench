import aligator
import aligator_bench_pywrap
import pinocchio as pin
import example_robot_data as erd
import numpy as np
import matplotlib.pyplot as plt

from aligator import constraints, manifolds
from pinocchio.visualize import MeshcatVisualizer

SEED = 42
pin.seed(SEED)
np.random.seed(SEED)

robot = erd.load("ur5")
rmodel: pin.Model = robot.model
nv = rmodel.nv
nq = rmodel.nq
nu = nv

EE_NAME = "tool0"

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

x0 = space.neutral()
dt = 0.01


def get_neutral_pose_tool_placement():
    q0 = pin.neutral(rmodel)
    rdata = rmodel.createData()
    ee_id = rmodel.getFrameId(EE_NAME)
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacement(rmodel, rdata, ee_id)
    return rdata.oMf[ee_id].copy()


M_tool_q0: pin.SE3 = get_neutral_pose_tool_placement()
print(M_tool_q0)
tool_dist_q0 = np.linalg.norm(M_tool_q0.translation)

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
times_ = np.linspace(0.0, tf, nsteps + 1)


def createTermConstraint(pf: np.ndarray):
    fn = aligator.FrameTranslationResidual(
        ndx, nu, rmodel, np.asarray(pf), rmodel.getFrameId(EE_NAME)
    )
    return aligator.StageConstraint(fn, constraints.EqualityConstraintSet())


problem = aligator.TrajOptProblem(x0, nu, space, term_cost)
pf = 0.95 * M_tool_q0.translation
problem.addTerminalConstraint(createTermConstraint(pf))

TOL = 1e-5
mu_init = 1e-4

for i in range(nsteps):
    stage = aligator.StageModel(rcost, dyn)
    _A = np.zeros((2 * nv, ndx))
    _b = np.empty(2 * nv)
    vmax = 10.0 * rmodel.velocityLimit
    vmin = -vmax
    _b[:nv] = -vmax  # v - vmax <= 0
    _A[:nv, nv:] = +np.eye(nv)

    _b[nv:] = +vmin  # vmin - v <= 0
    _A[nv:, nv:] = -np.eye(nv)

    vel_box_fn = aligator.LinearFunction(ndx, nu, ndx, 2 * nv)
    vel_box_fn.A[:] = _A
    vel_box_fn.d[:] = _b
    # stage.addConstraint(vel_box_fn, constraints.NegativeOrthant())
    problem.addStage(stage)


MAX_ITER = 400
alisolver = aligator.SolverProxDDP(TOL, mu_init, verbose=aligator.VERBOSE)
alisolver.max_iters = MAX_ITER
bcl_params: alisolver.AlmParams = alisolver.bcl_params
bcl_params.mu_lower_bound = 1e-10
alisolver.setup(problem)
alisolver.run(problem)

print(alisolver.results)

xs_ali = alisolver.results.xs.tolist()
us_ali = alisolver.results.us.tolist()

altro_solve = aligator_bench_pywrap.initAltroFromAligatorProblem(problem)
altro_opts = altro_solve.GetOptions()
altro_opts.verbose = aligator_bench_pywrap.AltroVerbosity.Inner
altro_opts.tol_cost = 1e-16
altro_opts.tol_stationarity = TOL
altro_opts.iterations_max = MAX_ITER
altro_opts.use_backtracking_linesearch = True

init_errcode = altro_solve.Initialize()
assert init_errcode == aligator_bench_pywrap.ErrorCodes.NoError
solve_code = altro_solve.Solve()

xs_altro = altro_solve.GetAllStates().tolist()
us_altro = altro_solve.GetAllInputs().tolist()

vs_ali = [x[nq:] for x in xs_ali]
fig1, axes1 = aligator.utils.plotting.plot_controls_traj(times_, us_ali)
fig2, axes2 = aligator.utils.plotting.plot_velocity_traj(times_, vs_ali, rmodel)
plt.show()
