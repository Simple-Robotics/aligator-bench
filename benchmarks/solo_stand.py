import pinocchio as pin
import numpy as np
import aligator
import matplotlib.pyplot as plt

from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer
from aligator import (
    constraints,
    manifolds,
    underactuatedConstrainedInverseDynamics,
    ControlErrorResidual,
    MultibodyFrictionConeResidual,
)
from aligator.utils import plotting
from .common import load_solo12, Args

robot = load_solo12()
rmodel: pin.Model = robot.model
rdata = rmodel.createData()


def get_default_pose():
    q0 = rmodel.referenceConfigurations["standing"].copy()
    fr_foot_fid = rmodel.getFrameId("FR_FOOT")
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacement(rmodel, rdata, fr_foot_fid)
    q0[2] -= rdata.oMf[fr_foot_fid].translation[2]
    q0[2] += 0.01
    pin.framesForwardKinematics(rmodel, rdata, q0)
    return q0


q_standing = get_default_pose()
q_up_stand = q_standing.copy()
torso_angle = np.deg2rad(-88)
q_up_stand[4] = torso_angle
q_up_stand[13] = 1.4
q_up_stand[14] = 0.2
q_up_stand[16] = 1.4
q_up_stand[17] = 0.2
hind_legs_idx = [13, 14, 16, 17]
# front legs
q_up_stand[7] = 1.7
q_up_stand[8] = -0.6
q_up_stand[10] = 1.7
q_up_stand[11] = -0.6
front_legs_idx = [7, 8, 10, 11]

FOOT_NAMES = ("FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT")
FOOT_NAMES_2 = ("HL_FOOT", "HR_FOOT")  # standing
FOOT_FRAME_IDS = {fname: rmodel.getFrameId(fname) for fname in FOOT_NAMES}
FOOT_JOINT_IDS = {
    fname: rmodel.frames[fid].parentJoint for fname, fid in FOOT_FRAME_IDS.items()
}


def get_constraint_models(feet_names, Kp, Kd=None):
    rcms = []
    if Kd is None:
        Kd = 2 * np.sqrt(Kp)

    for fname in feet_names:
        fid = FOOT_FRAME_IDS[fname]
        joint_id = FOOT_JOINT_IDS[fname]
        pl1 = rmodel.frames[fid].placement
        pl2 = rdata.oMf[fid]

        rcm = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            rmodel,
            joint_id,
            pl1,
            0,
            pl2,
            pin.LOCAL,
        )
        rcm.corrector.Kp[:] = Kp
        rcm.corrector.Kd[:] = Kd
        rcm.name = fname
        rcms.append(rcm)
    return rcms


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6  # not fully actuated
act_matrix = np.eye(nv, nu, -6)

prox_settings = pin.ProximalSettings(accuracy=1e-7, mu=1e-10, max_iter=12)
v0 = np.zeros(nv)
xref = np.concatenate((q_standing, v0))

tf = 4.0
dt = 0.01
nsteps = int(tf / dt)

space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx


def get_contact_dynamics_phase(feet_names, prox_settings):
    from aligator.dynamics import (
        MultibodyConstraintFwdDynamics,
        IntegratorSemiImplEuler,
    )

    rcm = get_constraint_models(feet_names, 1e3)
    ode = MultibodyConstraintFwdDynamics(space, act_matrix, rcm, prox_settings)
    dyn = IntegratorSemiImplEuler(ode, dt)
    return dyn, rcm


dyn1, rcms1 = get_contact_dynamics_phase(FOOT_NAMES, prox_settings)
dyn2, rcms2 = get_contact_dynamics_phase(FOOT_NAMES_2, prox_settings)  # standing

u0, _ = underactuatedConstrainedInverseDynamics(
    rmodel, rdata, q_standing, v0, act_matrix, rcms1, [cm.createData() for cm in rcms1]
)

t_stand = 0.9
t_up = 1.5
xref_up_stand = np.concatenate((q_up_stand, v0))

xs_init = []
us_init = [np.zeros(nu)] * nsteps
stages = []
for i in range(nsteps):
    ti = i * dt

    Wx = np.ones(ndx)
    Wx[:nv] = 1e-3
    Wx[:3] = 0.0
    _dyn = dyn1
    _target = xref.copy()
    rcost = aligator.CostStack(space, nu)
    w_fr_w = 1e-2
    if ti < t_stand:
        for feet in FOOT_NAMES:
            fr_cone = MultibodyFrictionConeResidual(
                ndx, rmodel, act_matrix, rcms1, prox_settings, feet, 0.01
            )
            w_fr = w_fr_w * np.eye(fr_cone.nr)
            rcost.addCost(
                f"fric_{feet}",
                aligator.QuadraticResidualCost(space, fr_cone, w_fr),
                dt,
            )
    elif ti >= t_stand:
        # switch to standing
        _dyn = dyn2
        _target[:] = xref
        _target[4] = torso_angle
        Wx[4] = 1.0
        for feet in FOOT_NAMES_2:
            fr_cone = MultibodyFrictionConeResidual(
                ndx, rmodel, act_matrix, rcms2, prox_settings, feet, 0.01
            )
            w_fr = w_fr_w * np.eye(fr_cone.nr)
            rcost.addCost(
                f"fric_{feet}",
                aligator.QuadraticResidualCost(space, fr_cone, w_fr),
                dt,
            )
    if ti >= t_up:
        Wx[hind_legs_idx] = 10.0
        Wx[front_legs_idx] = 1.0
        _target[:] = xref_up_stand

    xs_init.append(_target)

    rcost.addCost(
        "xreg", aligator.QuadraticStateCost(space, nu, _target, np.diag(Wx)), dt
    )
    rcost.addCost(
        "ureg", aligator.QuadraticControlCost(space, nu, np.eye(nu)), 1e-4 * dt
    )
    stage = aligator.StageModel(rcost, _dyn)
    umax = rmodel.effortLimit
    stage.addConstraint(
        ControlErrorResidual(ndx, nu), constraints.BoxConstraint(-umax, umax)
    )
    stages.append(stage)


Wx_term = 1e-2 * np.ones(ndx)
Wx_term[:3] = 0.0
Wx_term[4] = 1.0
Wx_term[hind_legs_idx] = 10.0
Wx_term[front_legs_idx] = 1.0
xs_init.append(xref_up_stand)
assert len(xs_init) == nsteps + 1
assert len(stages) == nsteps
term_cost = aligator.QuadraticStateCost(space, nu, xref_up_stand, np.diag(Wx_term))

problem = aligator.TrajOptProblem(xref, stages, term_cost)

if __name__ == "__main__":
    args = Args().parse_args()

    TOL = 1e-3
    MAX_ITER = 300
    match args.solver:
        case "ali":
            mu_init = 1e-2
            solver = aligator.SolverProxDDP(TOL, mu_init)
            solver.setup(problem)
            solver.verbose = aligator.VERBOSE
            solver.rollout_type = aligator.ROLLOUT_LINEAR
            solver.linesearch.avg_eta = 0.4
            solver.max_iters = MAX_ITER
            solver.run(problem, xs_init, us_init)

            print(solver.results)
            xs_opt = np.stack(solver.results.xs)
            print("xs_opt.shape:", xs_opt.shape)
            qs_opt = xs_opt[:, :nq]
            us_opt = np.stack(solver.results.us)
        case "ipopt":
            from aligator_bench_pywrap import SolverIpopt

            solver = SolverIpopt()
            solver.setup(problem)
            solver.setMaxIters(MAX_ITER)
            solver.setOption("tol", TOL)
            solver.solve()
            xs_opt = np.stack(solver.xs)
            qs_opt = xs_opt[:, :nq]
            us_opt = np.stack(solver.us)
        case _:
            raise ValueError("Solver not supported here")
    times = np.linspace(0, tf, nsteps + 1)

    plotting.plot_controls_traj(times, us_opt, rmodel=rmodel)
    plt.show()

    viz = MeshcatVisualizer(
        rmodel, visual_model=robot.visual_model, visual_data=robot.visual_data
    )
    viz.initViewer(open=True, loadModel=True)

    viz.display(q_standing)
    input("[enter]")

    for i in range(4):
        viz.play(qs_opt, dt=dt)
