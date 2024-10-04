import aligator
import aligator_bench_pywrap
import pinocchio as pin
import example_robot_data as erd
import numpy as np

from aligator import manifolds
from pinocchio.visualize import MeshcatVisualizer

SEED = 42
pin.seed(SEED)
np.random.seed(SEED)

robot = erd.load("ur5")
rmodel: pin.Model = robot.model
nv = rmodel.nv
nq = rmodel.nq

nu = nv

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

x0 = space.neutral()
x1 = space.rand()
dt = 0.01

rcost = aligator.CostStack(space, nu)
rcost.addCost(aligator.QuadraticStateCost(space, nu, x0, np.eye(ndx)), dt)
rcost.addCost(aligator.QuadraticControlCost(space, nu, 0.01 * np.eye(nu)), dt)
wx_term = 1e-4 * np.eye(ndx)
term_cost = aligator.QuadraticStateCost(space, nu, x1, wx_term)

dyn = aligator.dynamics.IntegratorSemiImplEuler(
    aligator.dynamics.MultibodyFreeFwdDynamics(space), dt
)

tf = 1.2
nsteps = int(tf / dt)

EE_NAME = "tool0"


def createConstraint(pf: np.ndarray):
    fn = aligator.FrameTranslationResidual(
        ndx, nu, rmodel, np.asarray(pf), rmodel.getFrameId(EE_NAME)
    )
    return aligator.StageConstraint(fn, aligator.constraints.EqualityConstraintSet())


problem = aligator.TrajOptProblem(x0, nu, space, term_cost)
problem.addTerminalConstraint(createConstraint([0.5, 0.5, 0.5]))

TOL = 1e-5
mu_init = 1.0

for i in range(nsteps):
    problem.addStage(aligator.StageModel(rcost, dyn))

alisolver = aligator.SolverProxDDP(TOL, mu_init, verbose=aligator.VERBOSE)
alisolver.setup(problem)
alisolver.run(problem)

print(alisolver.results)

xs_ali = alisolver.results.xs
us_ali = alisolver.results.us

altro_solve = aligator_bench_pywrap.initAltroFromAligatorProblem(problem)
altro_opts = altro_solve.GetOptions()
altro_opts.verbose = aligator_bench_pywrap.AltroVerbosity.Inner
altro_opts.tol_cost = 1e-16
altro_opts.tol_stationarity = TOL
altro_opts.use_backtracking_linesearch = True

init_errcode = altro_solve.Initialize()
assert init_errcode == aligator_bench_pywrap.ErrorCodes.NoError
solve_code = altro_solve.Solve()

xs_altro = altro_solve.GetAllStates()
us_altro = altro_solve.GetAllInputs()
