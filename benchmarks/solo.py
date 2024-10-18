import pinocchio as pin
import numpy as np
import aligator
import contextlib

from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer
from aligator.dynamics import MultibodyConstraintFwdDynamics, IntegratorSemiImplEuler
from aligator import constraints, manifolds, underactuatedConstrainedInverseDynamics
from .common import Args


args = Args().parse_args()


def load_solo12(verbose=False):
    """Load Solo12 with Euclidean parameterization."""
    from os.path import join
    from example_robot_data.robots_loader import (
        Solo12Loader,
        RobotWrapper,
        getModelPath,
        readParamsFromSrdf,
    )

    jmc = pin.JointModelComposite(pin.JointModelTranslation())
    jmc.addJoint(pin.JointModelSphericalZYX())
    df_path = join(
        Solo12Loader.path, Solo12Loader.urdf_subpath, Solo12Loader.urdf_filename
    )
    model_path = getModelPath(df_path, verbose)
    df_path = join(model_path, df_path)
    builder = RobotWrapper.BuildFromURDF
    robot = builder(df_path, [join(model_path, "../..")], jmc)
    srdf_path = join(
        model_path,
        Solo12Loader.path,
        Solo12Loader.srdf_subpath,
        Solo12Loader.srdf_filename,
    )
    robot.q0 = readParamsFromSrdf(
        robot.model,
        srdf_path,
        verbose,
        Solo12Loader.has_rotor_parameters,
        Solo12Loader.ref_posture,
    )

    return robot


robot = load_solo12()
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data


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

viz = MeshcatVisualizer(rmodel, visual_model=robot.visual_model, data=rdata)

FOOT_NAMES = ("FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT")
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

rcms1 = get_constraint_models(FOOT_NAMES, 1e3)
rcms2 = get_constraint_models(("FR_FOOT", "HL_FOOT", "HR_FOOT"), 1e3)

tf = 3.0
dt = 0.01
nsteps = int(tf / dt)
print(f"nsteps = {nsteps}")

prox_settings = pin.ProximalSettings(1e-7, 1e-10, 10)
ode1 = MultibodyConstraintFwdDynamics(space, act_matrix, rcms1, prox_settings)
ode2 = MultibodyConstraintFwdDynamics(space, act_matrix, rcms2, prox_settings)
v0 = np.zeros(nv)
xref = np.concatenate((q_standing, v0))
u0, _ = underactuatedConstrainedInverseDynamics(
    rmodel, rdata, q_standing, v0, act_matrix, rcms1, [cm.createData() for cm in rcms1]
)

dyn1 = IntegratorSemiImplEuler(ode1, dt)
dyn2 = IntegratorSemiImplEuler(ode2, dt)

xs_init = [xref] * (nsteps + 1)
us_init = [u0] * nsteps
print(f"xref = {xref}")


def create_target(ti: float):
    xtgt = xref.copy()
    freq = 3.0
    phas = freq * ti
    amp = xtgt[2] * 1.1
    xtgt[2] += amp * np.sin(phas)

    if 1.0 <= ti < 1.5:
        xtgt[4] = np.deg2rad(30)
    if 1.5 <= ti:
        xtgt[4] = np.deg2rad(-20)

    return xtgt


def ee_fl_foot_xres():
    from aligator import FrameTranslationResidual

    fid = FOOT_FRAME_IDS["FL_FOOT"]
    fres = FrameTranslationResidual(ndx, nu, rmodel, np.zeros(3), fid)[0]
    return fres


stages = []
for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)
    ti = dt * i

    _target = create_target(ti)
    _dyn = dyn1
    if ti >= 2.0:
        _dyn = dyn2
    Wx = 1e-2 * np.ones(ndx)
    Wx[[2, 4]] = 2.0
    Wx = np.diag(Wx)

    rcost.addCost("xreg", aligator.QuadraticStateCost(space, nu, _target, Wx), dt)
    rcost.addCost(
        "ureg", aligator.QuadraticControlCost(space, nu, np.eye(nu)), 1e-1 * dt
    )
    stage = aligator.StageModel(rcost, _dyn)
    if ti >= 2.5:
        fres = ee_fl_foot_xres()
        stage.addConstraint(fres, constraints.NegativeOrthant())
    stages.append(stage)

xterm = xref.copy()
Wx_term = 1e-1 * np.ones(ndx)
Wx_term[[4, 5]] = 4.0
xterm[4] = np.deg2rad(-25)
xterm[5] = np.deg2rad(20)
term_cost = aligator.QuadraticStateCost(space, nu, xterm, np.diag(Wx_term))
problem = aligator.TrajOptProblem(xref, stages, term_cost)


TOL = 1e-3
MAX_ITER = 200
match args.solver:
    case "ali":
        mu_init = 1e-4
        solver = aligator.SolverProxDDP(TOL, mu_init)
        solver.rollout_type = aligator.ROLLOUT_LINEAR
        solver.verbose = aligator.VERBOSE
        solver.max_iters = MAX_ITER
        solver.setup(problem)
        # solver.run(problem, xs_init, us_init)
        solver.run(problem)

        results = solver.results
        print(results)
        xs_opt = results.xs
    case "ipopt":
        from aligator_bench_pywrap import SolverIpopt

        solver = SolverIpopt()
        solver.setup(problem, True)
        # solver.setInitialGuess(xs_init, us_init)
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

viz.setCameraPosition([0.5, 0.5, 0.5])
ctx = (
    viz.create_video_ctx("solo12_lift_paw.mp4")
    if args.record
    else contextlib.nullcontext()
)
with ctx:
    viz.play(qs_opt, dt=dt)
# viz.display(q_standing)
