import pinocchio as pin
import numpy as np
import aligator

from example_robot_data.robots_loader import BoltLoader, RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from aligator.dynamics import MultibodyConstraintFwdDynamics, IntegratorSemiImplEuler
from aligator import manifolds
from aligator import underactuatedConstrainedInverseDynamics


class BoltEuclidean(BoltLoader):
    free_flyer = False


def load_bolt(verbose=False):
    from os.path import join
    from example_robot_data.robots_loader import getModelPath, readParamsFromSrdf

    jmc = pin.JointModelComposite(pin.JointModelTranslation())
    jmc.addJoint(pin.JointModelSphericalZYX())
    df_path = join(
        BoltEuclidean.path, BoltEuclidean.urdf_subpath, BoltEuclidean.urdf_filename
    )
    model_path = getModelPath(df_path, verbose)
    df_path = join(model_path, df_path)
    builder = RobotWrapper.BuildFromURDF
    robot = builder(df_path, [join(model_path, "../..")], jmc)
    srdf_path = join(
        model_path,
        BoltEuclidean.path,
        BoltEuclidean.srdf_subpath,
        BoltEuclidean.srdf_filename,
    )
    robot.q0 = readParamsFromSrdf(
        robot.model,
        srdf_path,
        verbose,
        BoltEuclidean.has_rotor_parameters,
        BoltEuclidean.ref_posture,
    )

    return robot


robot = load_bolt()
rmodel: pin.Model = robot.model
rdata: pin.Data = robot.data

q_standing = rmodel.referenceConfigurations["standing"]
fl_foot_fid = rmodel.getFrameId("FL_FOOT")
fr_foot_fid = rmodel.getFrameId("FR_FOOT")
pin.forwardKinematics(rmodel, rdata, q_standing)
pin.updateFramePlacement(rmodel, rdata, fr_foot_fid)
q_standing[2] -= rdata.oMf[fr_foot_fid].translation[2]
pin.framesForwardKinematics(rmodel, rdata, q_standing)

viz = MeshcatVisualizer(rmodel, visual_model=robot.visual_model, data=rdata)

FOOT_FRAME_IDS = {fname: rmodel.getFrameId(fname) for fname in ("FL_FOOT", "FR_FOOT")}

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
        rcms.append(rcm)
    return rcms


nq = rmodel.nq
nv = rmodel.nv
nu = nv - 6  # not fully actuated
act_matrix = np.eye(nv, nu, -6)
space = manifolds.MultibodyPhaseSpace(rmodel)
ndx = space.ndx

rcms1 = get_constraint_models(("FL_FOOT", "FR_FOOT"))
ode1 = MultibodyConstraintFwdDynamics(
    space,
    act_matrix,
    rcms1,
    pin.ProximalSettings(1e-6, 1e-10, 10),
)

nsteps = 200
dt = 20e-3

xref = np.concatenate((q_standing, np.zeros(nv)))

dyn1 = IntegratorSemiImplEuler(ode1, dt)
rcost = aligator.CostStack(space, nu)
rcost.addCost(
    "xreg", aligator.QuadraticStateCost(space, nu, xref, dt * np.eye(ndx)), 1e-1
)
rcost.addCost("ureg", aligator.QuadraticControlCost(space, nu, dt * np.eye(nu)), 1e-2)

stage = aligator.StageModel(rcost, dyn1)
stages = [stage] * nsteps

term_cost = aligator.QuadraticStateCost(space, nu, xref, np.eye(ndx))
problem = aligator.TrajOptProblem(xref, stages, term_cost)

u0_stable, _ = underactuatedConstrainedInverseDynamics(
    rmodel,
    rdata,
    q_standing,
    np.zeros(nv),
    act_matrix,
    rcms1,
    [cm.createData() for cm in rcms1],
)
us_init = [u0_stable] * nsteps
xs_init = [xref] * (nsteps + 1)


TOL = 1e-3
MAX_ITER = 400
mu_init = 1e-4
solver = aligator.SolverProxDDP(TOL, mu_init)
solver.rollout_type = aligator.ROLLOUT_LINEAR
solver.verbose = aligator.VERBOSE
solver.max_iters = MAX_ITER
solver.setup(problem)
solver.run(problem, xs_init, us_init)

results = solver.results
# print(results)


viz.initViewer(open=True, loadModel=True)

qs_opt = np.asarray(results.xs)[:, :nq]
print(qs_opt)

input("[enter]")

viz.play(qs_opt, dt=2 * dt)
