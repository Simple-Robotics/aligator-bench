"""
Modified version of aligator/examples/ur10_ballistic.py:
https://github.com/Simple-Robotics/aligator/blob/devel/examples/ur10_ballistic.py
"""

import example_robot_data as erd
import pinocchio as pin
import numpy as np
import aligator
import hppfcl
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple
from pinocchio.visualize import MeshcatVisualizer
from aligator.utils.plotting import (
    plot_controls_traj,
    plot_convergence,
    plot_velocity_traj,
)
from .common import add_namespace_prefix_to_models, Args as _Args
from aligator import dynamics, manifolds, constraints


class Args(_Args):
    pass


Args.solver = "ali"


def load_projectile_model(free_flyer: bool = True):
    ball_urdf = Path("assets") / "mug.urdf"
    packages_dirs = [ball_urdf.parent]
    ball_scale = 1.0
    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        str(ball_urdf),
        package_dirs=packages_dirs,
        root_joint=pin.JointModelFreeFlyer()
        if free_flyer
        else pin.JointModelTranslation(),
    )
    for geom in cmodel.geometryObjects:
        geom.meshScale *= ball_scale
    for geom in vmodel.geometryObjects:
        geom.meshScale *= ball_scale
    return model, cmodel, vmodel


def append_ball_to_robot_model(
    robot: pin.RobotWrapper,
) -> Tuple[pin.Model, pin.GeometryModel, pin.GeometryModel]:
    base_model: pin.Model = robot.model
    base_visual: pin.GeometryModel = robot.visual_model
    base_coll: pin.GeometryModel = robot.collision_model
    ee_link_id = base_model.getFrameId("tool0")
    _ball_model, _ball_coll, _ball_visu = load_projectile_model(free_flyer=False)
    add_namespace_prefix_to_models(_ball_model, _ball_coll, _ball_visu, "ball")

    pin.forwardKinematics(base_model, robot.data, robot.q0)
    pin.updateFramePlacement(base_model, robot.data, ee_link_id)

    tool_frame_pl = robot.data.oMf[ee_link_id]
    rel_placement = tool_frame_pl.copy()
    rel_placement.translation[1] = 0.0
    rmodel, cmodel = pin.appendModel(
        base_model, _ball_model, base_coll, _ball_coll, 0, rel_placement
    )
    _, vmodel = pin.appendModel(
        base_model, _ball_model, base_visual, _ball_visu, 0, rel_placement
    )

    ref_q0 = pin.neutral(rmodel)
    ref_q0[: base_model.nq] = robot.q0
    return rmodel, cmodel, vmodel, ref_q0


def create_rcm(rmodel, rdata, q0, contact_type=pin.ContactType.CONTACT_6D):
    # create rigid constraint between ball & tool0
    tool_fid = rmodel.getFrameId("tool0")
    frame: pin.Frame = rmodel.frames[tool_fid]
    joint1_id = frame.parentJoint
    joint2_id = rmodel.getJointId("ball/root_joint")
    pin.framesForwardKinematics(rmodel, rdata, q0)
    pl1 = rmodel.frames[tool_fid].placement
    pl2 = rdata.oMf[tool_fid]
    rcm = pin.RigidConstraintModel(
        contact_type,
        rmodel,
        joint1_id,
        pl1,
        joint2_id,
        pl2,
        pin.LOCAL_WORLD_ALIGNED,
    )
    Kp = 1e-3
    rcm.corrector.Kp[:] = Kp
    rcm.corrector.Kd[:] = 2 * Kp**0.5
    return rcm


class URBallistic(object):
    robot = erd.load("ur10")
    rmodel, cmodel, vmodel, ref_q0 = append_ball_to_robot_model(robot)

    def __init__(self, target_pos):
        q0_ref_arm = np.array(
            [0.0, np.deg2rad(-120), 2 * np.pi / 3, np.deg2rad(-45), 0.0, 0.0]
        )
        self.robot.q0[:] = q0_ref_arm
        self._debug = False
        if self._debug:
            print(f"Velocity limit (before): {self.robot.model.velocityLimit}")
            print(f"New model velocity lims: {self.rmodel.velocityLimit}")

        self.rdata: pin.Data = self.rmodel.createData()
        self.nq_o = self.robot.model.nq
        self.nv_o = self.robot.model.nv
        self.nq_b = self.rmodel.nq
        self.nv_b = self.rmodel.nv

        self._build_problem(target_pos)

    @staticmethod
    def name():
        return "UR10_Ballistic"

    def _build_problem(self, target_pos):
        space = manifolds.MultibodyPhaseSpace(self.rmodel)
        nu = self.nv_o
        ndx = space.ndx
        x0 = space.neutral()
        x0[: self.nq_b] = self.ref_q0
        MUG_VEL_IDX = slice(self.robot.nv, self.nv_b)
        JOINT_VEL_LIM_IDX = [0, 1, 3, 4, 5, 6]

        self.dt = 0.01
        self.tf = 2.0  # seconds
        self.nsteps = int(self.tf / self.dt)
        actuation_matrix = np.eye(self.nv_b, nu)

        prox_settings = pin.ProximalSettings(accuracy=1e-8, mu=1e-6, max_iter=20)
        rcm = create_rcm(
            self.rmodel,
            self.rdata,
            q0=self.ref_q0,
            contact_type=pin.ContactType.CONTACT_3D,
        )
        ode1 = dynamics.MultibodyConstraintFwdDynamics(
            space, actuation_matrix, [rcm], prox_settings
        )
        ode2 = dynamics.MultibodyFreeFwdDynamics(space, actuation_matrix)
        dyn_model1 = dynamics.IntegratorSemiImplEuler(ode1, self.dt)
        dyn_model2 = dynamics.IntegratorSemiImplEuler(ode2, self.dt)

        q0 = x0[: self.nq_b]
        v0 = x0[self.nq_b :]
        u0_free = pin.rnea(
            self.robot.model,
            self.robot.data,
            self.robot.q0,
            self.robot.v0,
            self.robot.v0,
        )
        u0, lam_c = aligator.underactuatedConstrainedInverseDynamics(
            self.rmodel, self.rdata, q0, v0, actuation_matrix, [rcm], [rcm.createData()]
        )
        assert u0.shape == (nu,)

        def testu0(u0):
            pin.initConstraintDynamics(self.rmodel, self.rdata, [rcm])
            rcd = rcm.createData()
            tau = actuation_matrix @ u0
            acc = pin.constraintDynamics(
                self.rmodel, self.rdata, q0, v0, tau, [rcm], [rcd]
            )
            print("plugging in u0, got acc={}".format(acc))

        if self._debug:
            with np.printoptions(precision=4, linewidth=200):
                print("invdyn (free): {}".format(u0_free))
                print("invdyn torque : {}".format(u0))
                testu0(u0)

        dms = [dyn_model1] * self.nsteps
        us_i = [u0] * len(dms)
        xs_i = aligator.rollout(dms, x0, us_i)

        def create_running_cost():
            costs = aligator.CostStack(space, nu)
            w_x = np.array([1e-3] * self.nv_b + [0.1] * self.nv_b)
            w_v = w_x[self.nv_b :]
            # no costs on mug
            w_x[MUG_VEL_IDX] = 0.0
            w_v[MUG_VEL_IDX] = 0.0
            assert space.isNormalized(x0)
            xreg = aligator.QuadraticStateCost(space, nu, x0, np.diag(w_x) * self.dt)
            w_u = np.ones(nu) * 1e-5
            ureg = aligator.QuadraticControlCost(space, u0, np.diag(w_u) * self.dt)
            costs.addCost(xreg)
            costs.addCost(ureg)
            return costs

        def create_term_cost(has_frame_cost=False, w_ball=1.0):
            w_xf = np.zeros(ndx)
            w_xf[: self.robot.nv] = 1e-4
            w_xf[self.nv_o :] = 1e-6
            costs = aligator.CostStack(space, nu)
            xreg = aligator.QuadraticStateCost(space, nu, x0, np.diag(w_xf))
            costs.addCost(xreg)
            if has_frame_cost:
                ball_pos_fn = get_ball_fn(target_pos)
                w_ball = np.eye(ball_pos_fn.nr) * w_ball
                ball_cost = aligator.QuadraticResidualCost(space, ball_pos_fn, w_ball)
                costs.addCost(ball_cost)
            return costs

        def get_ball_fn(target_pos):
            fid = self.rmodel.getFrameId("ball/root_joint")
            return aligator.FrameTranslationResidual(
                ndx, nu, self.rmodel, target_pos, fid
            )

        def create_term_constraint(target_pos):
            term_fn = get_ball_fn(target_pos)
            return (term_fn, constraints.EqualityConstraintSet())

        def get_position_limit_constraint():
            state_fn = aligator.StateErrorResidual(space, nu, space.neutral())
            pos_fn = state_fn[:7]
            box_cstr = constraints.BoxConstraint(
                self.robot.model.lowerPositionLimit, self.robot.model.upperPositionLimit
            )
            return (pos_fn, box_cstr)

        def get_velocity_limit_constraint():
            state_fn = aligator.StateErrorResidual(space, nu, space.neutral())
            vel_fn = state_fn[[self.nv_b + i for i in JOINT_VEL_LIM_IDX]]
            vlim = self.rmodel.velocityLimit[JOINT_VEL_LIM_IDX]
            assert vel_fn.nr == vlim.shape[0]
            box_cstr = constraints.BoxConstraint(-vlim, vlim)
            return (vel_fn, box_cstr)

        def get_torque_limit_constraint():
            ctrlfn = aligator.ControlErrorResidual(ndx, np.zeros(nu))
            eff = self.robot.model.effortLimit
            box_cstr = constraints.BoxConstraint(-eff, eff)
            return (ctrlfn, box_cstr)

        def create_stage(contact: bool):
            dm = dyn_model1 if contact else dyn_model2
            rc = create_running_cost()
            stm = aligator.StageModel(rc, dm)
            stm.addConstraint(*get_torque_limit_constraint())
            # stm.addConstraint(get_position_limit_constraint())
            stm.addConstraint(*get_velocity_limit_constraint())
            return stm

        stages = []
        self.t_contact = int(0.4 * self.nsteps)
        for k in range(self.nsteps):
            stages.append(create_stage(k <= self.t_contact))

        term_cost = create_term_cost()

        problem = aligator.TrajOptProblem(x0, stages, term_cost)
        problem.addTerminalConstraint(*create_term_constraint(target_pos))
        problem.addTerminalConstraint(*get_velocity_limit_constraint())
        self.problem = problem
        self.xs_i = xs_i
        self.us_i = us_i


if __name__ == "__main__":
    target_pos = np.array([2.4, -0.2, 0.8])
    example = URBallistic(target_pos=target_pos)
    robot_o = example.robot
    problem = example.problem
    rmodel = example.rmodel
    rdata = example.rdata
    dt = example.dt
    tf = example.tf
    nsteps = example.nsteps

    xs_i = example.xs_i
    us_i = example.us_i

    nq_b = example.nq_b
    nv_b = example.nv_b
    nq_o = example.nq_o
    nv_o = example.nv_o

    qs_i = [x[:nq_b] for x in xs_i]

    def configure_viz(target_pos):
        gobj = pin.GeometryObject(
            "objective", 0, pin.SE3(np.eye(3), target_pos), hppfcl.Sphere(0.04)
        )
        gobj.meshColor[:] = np.array([200, 100, 100, 200]) / 255.0

        viz = MeshcatVisualizer(
            model=example.rmodel,
            collision_model=example.cmodel,
            visual_model=example.vmodel,
            data=rdata,
        )
        viz.initViewer(open=True, loadModel=True)
        viz.addGeometryObject(gobj)
        viz.setCameraZoom(1.7)
        return viz

    args = Args().parse_args()

    if args.viz:
        viz = configure_viz(target_pos=target_pos)
        viz.play(qs_i, dt=dt)
    else:
        viz = None

    TOL = 1e-4
    MAX_ITERS = 300
    if args.solver == "ali":
        mu_init = 2e-1
        solver = aligator.SolverProxDDP(
            TOL, mu_init, max_iters=MAX_ITERS, verbose=aligator.VERBOSE
        )
        his_cb = aligator.HistoryCallback(solver)
        solver.registerCallback("his", his_cb)
        solver.setup(problem)
        flag = solver.run(problem, xs_i, us_i)

        print(solver.results)
        ws: aligator.Workspace = solver.workspace
        rs: aligator.Results = solver.results

        xs_opt = np.asarray(solver.results.xs)
        us_opt = np.asarray(solver.results.us)
    elif args.solver == "ipopt":
        from aligator_bench_pywrap import SolverIpopt

        solver = SolverIpopt()
        solver.setup(problem, True)
        solver.setAbsTol(TOL)
        solver.setOption("hessian_approximation", "limited-memory")
        solver.setMaxIters(MAX_ITERS)
        solver.solve()
        xs_opt = np.asarray(solver.xs)
        us_opt = np.asarray(solver.us)

    qs_opt = xs_opt[:, :nq_b]
    vs_opt = xs_opt[:, nq_b:]
    proj_frame_id = rmodel.getFrameId("ball/root_joint")

    def get_frame_vel(k: int):
        pin.forwardKinematics(rmodel, rdata, qs_opt[k], vs_opt[k])
        return pin.getFrameVelocity(rmodel, rdata, proj_frame_id)

    vf_before_launch = get_frame_vel(example.t_contact)
    vf_launch_t = get_frame_vel(example.t_contact + 1)
    print("Before launch  :", vf_before_launch.np)
    print("Launch velocity:", vf_launch_t.np)

    EXPERIMENT_NAME = "ur10_mug_throw"

    if args.viz:

        def viz_callback(i: int):
            pin.forwardKinematics(rmodel, rdata, qs_opt[i], xs_opt[i][nq_b:])
            viz.drawFrameVelocities(proj_frame_id, v_scale=0.06)
            fid = rmodel.getFrameId("ball/root_joint")
            ctar: pin.SE3 = rdata.oMf[fid]
            viz.setCameraTarget(ctar.translation)

        input("[press enter]")

        viz.play(qs_opt, dt, callback=viz_callback)

    if args.plot:
        times = np.linspace(0.0, tf, nsteps + 1)
        _joint_names = robot_o.model.names
        _efflims = robot_o.model.effortLimit
        _vlims = robot_o.model.velocityLimit
        figsize = (6.4, 4.0)
        fig1, _ = plot_controls_traj(
            times, us_opt, rmodel=rmodel, effort_limit=_efflims, figsize=figsize
        )
        fig1.suptitle("Controls (N/m)")
        fig2, _ = plot_velocity_traj(
            times,
            vs_opt[:, : nv_o - nv_b],
            rmodel=robot_o.model,
            vel_limit=_vlims,
            figsize=figsize,
        )

        PLOTDIR = Path("results")
        _fig_dict = {"controls": fig1, "velocity": fig2}

        if args.solver == "ali":
            fig4 = plt.figure(figsize=(6.4, 3.6))
            ax = fig4.add_subplot(111)
            plot_convergence(
                his_cb,
                ax,
                res=solver.results,
                show_al_iters=True,
                legend_kwargs=dict(fontsize=8),
            )
            ax.set_title("Convergence")
            fig4.tight_layout()
            _fig_dict["conv"] = fig4

            for name, fig in _fig_dict.items():
                figpath: Path = PLOTDIR / f"{EXPERIMENT_NAME}_{name}"
                fig.savefig(figpath.with_suffix(".pdf"))

        plt.show()
