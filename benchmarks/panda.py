import aligator
import pinocchio as pin
import numpy as np
import example_robot_data as erd

from aligator import manifolds


robot = erd.load("panda")
rmodel: pin.Model = robot.model
rdata = rmodel.createData()

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx
nu = rmodel.nv

x0 = space.neutral()
term_cost = aligator.CostStack(space, nu)
term_cost.addCost(aligator.QuadraticStateCost(space, nu, x0, np.eye(ndx)))
ee_id = rmodel.getFrameId("panda_rightfinger")
ee_ref = np.array([0.0, 0.5, 0.5])
w_ee = np.eye(3) * 10.0
term_cost.addCost(
    aligator.QuadraticResidualCost(
        space, aligator.FrameTranslationResidual(ndx, nu, rmodel, ee_ref, ee_id), w_ee
    )
)

nsteps = 100
dt = 0.05
tf = nsteps * dt

ode = aligator.dynamics.MultibodyFreeFwdDynamics(space)
dynamics = aligator.dynamics.IntegratorSemiImplEuler(ode, dt)
rcost = aligator.CostStack(space, nu)
rcost.addCost("xreg", aligator.QuadraticStateCost(space, nu, x0, dt * np.eye(ndx)))
rcost.addCost(
    "ureg", aligator.QuadraticControlCost(space, np.zeros(nu), dt * np.eye(nu))
)

stage = aligator.StageModel(rcost, dynamics)

stages = []
for i in range(nsteps):
    stages.append(stage)

problem = aligator.TrajOptProblem(x0, stages, term_cost)

TOL = 1e-5
mu_init = 1e-3
solver = aligator.SolverProxDDP(TOL, mu_init, verbose=aligator.VERBOSE)
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.setup(problem)
ret = solver.run(problem)
print(ret)

# solver = SolverIpopt()
# solver.setOption("tol", TOL)
# solver.setup(problem)
# ret = solver.solve()
# print(ret)
