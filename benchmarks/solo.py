import pinocchio as pin
import numpy as np
import aligator

from pinocchio.visualize import MeshcatVisualizer
from aligator.dynamics import MultibodyConstraintFwdDynamics, IntegratorSemiImplEuler
from aligator import manifolds
from .common import Args


args = Args().parse_args()


def load_solo8(verbose=False):
    """Load Solo8 with Euclidean parameterization."""
    from os.path import join
    from example_robot_data.robots_loader import (
        Solo8Loader,
        RobotWrapper,
        getModelPath,
        readParamsFromSrdf,
    )

    jmc = pin.JointModelComposite(pin.JointModelTranslation())
    jmc.addJoint(pin.JointModelSphericalZYX())
    df_path = join(
        Solo8Loader.path, Solo8Loader.urdf_subpath, Solo8Loader.urdf_filename
    )
    model_path = getModelPath(df_path, verbose)
    df_path = join(model_path, df_path)
    builder = RobotWrapper.BuildFromURDF
    robot = builder(df_path, [join(model_path, "../..")], jmc)
    srdf_path = join(
        model_path,
        Solo8Loader.path,
        Solo8Loader.srdf_subpath,
        Solo8Loader.srdf_filename,
    )
    robot.q0 = readParamsFromSrdf(
        robot.model,
        srdf_path,
        verbose,
        Solo8Loader.has_rotor_parameters,
        Solo8Loader.ref_posture,
    )

    return robot


robot = load_solo8()
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


def get_constraint_models(feet_names, Kp=(0.0, 0.0, 1e2), Kd=50):
    rcms = []

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

rcms1 = get_constraint_models(FOOT_NAMES)
rcms2 = get_constraint_models([])

nsteps = 200
dt = 0.01
tf = nsteps * dt

prox_settings = pin.ProximalSettings(1e-8, 1e-10, 10)
ode1 = MultibodyConstraintFwdDynamics(space, act_matrix, rcms1, prox_settings)
ode2 = MultibodyConstraintFwdDynamics(space, act_matrix, rcms2, prox_settings)
xref = np.concatenate((q_standing, np.zeros(nv)))

dyn1 = IntegratorSemiImplEuler(ode1, dt)
dyn2 = IntegratorSemiImplEuler(ode2, dt)

n0 = 80
n1 = 110


stages = []
for i in range(nsteps):
    rcost = aligator.CostStack(space, nu)

    _target = xref.copy()
    _dyn = dyn1
    # if n0 <= i < n1:
    #     _dyn = dyn2
    #     _target[2] = 0.3
    Wx = np.diag(np.ones(ndx))

    rcost.addCost("xreg", aligator.QuadraticStateCost(space, nu, _target, dt * Wx), 1.0)
    rcost.addCost(
        "ureg", aligator.QuadraticControlCost(space, nu, dt * np.eye(nu)), 1e-1
    )
    stage = aligator.StageModel(rcost, _dyn)
    stages.append(stage)

Wx_term = 1e-3 * np.eye(ndx)
term_cost = aligator.QuadraticStateCost(space, nu, xref, Wx_term)
problem = aligator.TrajOptProblem(xref, stages, term_cost)


TOL = 1e-3
MAX_ITER = 200
match args.solver:
    case "ali":
        mu_init = 1e-4
        solver = aligator.SolverProxDDP(TOL, mu_init)
        solver.verbose = aligator.VERBOSE
        solver.max_iters = MAX_ITER
        solver.setup(problem)
        solver.run(problem)

        results = solver.results
        print(results)
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

viz.play(qs_opt, dt=dt)
# viz.display(q_standing)
