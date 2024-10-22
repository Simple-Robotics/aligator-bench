import pinocchio as pin
import numpy as np
import aligator

from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer
from aligator.dynamics import MultibodyConstraintFwdDynamics, IntegratorSemiImplEuler
from aligator import manifolds
from .common import Args


args = Args().parse_args()


def load_bolt(verbose=False):
    from os.path import join
    from example_robot_data.robots_loader import (
        BoltLoader,
        RobotWrapper,
        getModelPath,
        readParamsFromSrdf,
    )

    jmc = pin.JointModelComposite(pin.JointModelTranslation())
    jmc.addJoint(pin.JointModelSphericalZYX())
    df_path = join(BoltLoader.path, BoltLoader.urdf_subpath, BoltLoader.urdf_filename)
    model_path = getModelPath(df_path, verbose)
    df_path = join(model_path, df_path)
    builder = RobotWrapper.BuildFromURDF
    robot = builder(df_path, [join(model_path, "../..")], jmc)
    srdf_path = join(
        model_path,
        BoltLoader.path,
        BoltLoader.srdf_subpath,
        BoltLoader.srdf_filename,
    )
    robot.q0 = readParamsFromSrdf(
        robot.model,
        srdf_path,
        verbose,
        BoltLoader.has_rotor_parameters,
        BoltLoader.ref_posture,
    )

    return robot


robot = load_bolt()
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data


def get_default_pose():
    q0 = rmodel.referenceConfigurations["standing"].copy()
    fr_foot_fid = rmodel.getFrameId("FR_FOOT")
    jid0 = rmodel.getJointId("FL_HAA")
    jid1 = rmodel.getJointId("FR_HAA")
    id0 = rmodel.idx_qs[jid0]
    id1 = rmodel.idx_qs[jid1]
    q0[id0] *= 0.2
    q0[id1] *= 0.2
    pin.forwardKinematics(rmodel, rdata, q0)
    pin.updateFramePlacement(rmodel, rdata, fr_foot_fid)
    q0[2] -= rdata.oMf[fr_foot_fid].translation[2]
    q0[2] += 0.01
    return q0.copy()


q_standing = get_default_pose()

viz = MeshcatVisualizer(rmodel, visual_model=robot.visual_model, data=rdata)

FOOT_NAMES = ("FL_ANKLE", "FR_ANKLE")
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
            pin.LOCAL_WORLD_ALIGNED,
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
space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

pin.framesForwardKinematics(rmodel, rdata, q_standing)
rcms1 = get_constraint_models(FOOT_NAMES, 1e3)
rcms2 = get_constraint_models([], 1e3)

nsteps = 200
dt = 20e-3
tf = nsteps * dt

prox_settings = pin.ProximalSettings(1e-7, 1e-10, 10)
ode1 = MultibodyConstraintFwdDynamics(space, act_matrix, rcms1, prox_settings)
ode2 = MultibodyConstraintFwdDynamics(space, act_matrix, rcms2, prox_settings)
xref = np.concatenate((q_standing, np.zeros(nv)))

dyn1 = IntegratorSemiImplEuler(ode1, dt)
dyn2 = IntegratorSemiImplEuler(ode2, dt)

n0 = 80
n1 = 110


def get_target(i: int):
    xtgt = xref.copy()
    if n0 <= i < n1:
        xtgt[2] *= 0.5
    return xtgt


stages = []
for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)

    Wx = np.ones(ndx)
    Wx = np.diag(Wx)
    _target = get_target(i)
    _dyn = dyn1

    rcost.addCost("xreg", aligator.QuadraticStateCost(space, nu, _target, dt * Wx), 1.0)
    rcost.addCost(
        "ureg", aligator.QuadraticControlCost(space, nu, dt * np.eye(nu)), 1e-6
    )
    grav_res = aligator.GravityCompensationResidual(ndx, act_matrix, rmodel)
    assert grav_res.nr == rmodel.nv
    u_grav_cost = aligator.QuadraticResidualCost(
        space,
        grav_res,
        dt * np.eye(grav_res.nr),
    )
    rcost.addCost("ureg", u_grav_cost, 1e-1)
    stage = aligator.StageModel(rcost, _dyn)
    stages.append(stage)

term_cost = aligator.QuadraticStateCost(space, nu, xref, np.eye(ndx))
problem = aligator.TrajOptProblem(xref, stages, term_cost)


TOL = 1e-3
MAX_ITER = 400
match args.solver:
    case "ali":
        mu_init = 1e-4
        solver = aligator.SolverProxDDP(TOL, mu_init)
        solver.rollout_type = aligator.ROLLOUT_LINEAR
        solver.verbose = aligator.VERBOSE
        solver.max_iters = MAX_ITER
        solver.setup(problem)
        solver.run(problem)

        results = solver.results
        # print(results)
        xs_opt = results.xs
    case "ipopt":
        from aligator_bench_pywrap import SolverIpopt

        solver = SolverIpopt()
        solver.setup(problem, True)
        solver.setOption("tol", TOL)
        solver.setMaxIters(MAX_ITER)
        solver.solve()
        xs_opt = solver.xs.tolist()
    case "altro":
        from aligator_bench_pywrap import (
            ALTROSolver,
            AltroVerbosity,
            initAltroFromAligatorProblem,
        )

        _altrosolver: ALTROSolver = initAltroFromAligatorProblem(problem)
        init_code = _altrosolver.Initialize()
        _altro_opts = _altrosolver.GetOptions()
        _altro_opts.iterations_max = MAX_ITER
        _altro_opts.tol_cost = 1e-16
        _altro_opts.tol_primal_feasibility = TOL
        _altro_opts.tol_stationarity = TOL
        _altro_opts.verbose = AltroVerbosity.Inner
        _altro_opts.use_backtracking_linesearch = True
        _altrosolver.Solve()
        xs_opt = _altrosolver.GetAllStates().tolist()

viz.initViewer(open=True, loadModel=True)

qs_opt = np.asarray(xs_opt)[:, :nq]

input("[enter]")

viz.setCameraPosition([1.0, 1.0, 0.6])
viz.play(qs_opt, dt=dt)
