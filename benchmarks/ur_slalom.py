import aligator
import numpy as np
import pinocchio as pin

from aligator import manifolds, constraints
from aligator_bench_pywrap import SphereCylinderCollisionDistance
from .common import Args

import example_robot_data as erd
import hppfcl as fcl

default_p_ee_term = np.array([0.0, -0.4, 0.3])
default_cyl1_center = np.array([+0.5, -0.3, 0.0])


class UrSlalomExample(object):
    robot = erd.load("ur5")
    rmodel: pin.Model = robot.model
    rdata: pin.Data = robot.data
    coll_model: pin.GeometryModel = robot.collision_model
    vis_model: pin.GeometryModel = robot.visual_model

    def __init__(self, q0=robot.q0, p_ee_term=None, cyl1_center=None):
        if cyl1_center is None:
            cyl1_center = default_cyl1_center.copy()
        if p_ee_term is None:
            p_ee_term = default_p_ee_term.copy()
        self.problem, self.xs_init, self.us_init = self._build_problem(
            q0=q0, cyl1_center=cyl1_center, p_ee_term=np.asarray(p_ee_term)
        )

    @staticmethod
    def name():
        return "UR5_Collision"

    def add_objective_viz(self, name, pos):
        sph = fcl.Sphere(0.02)
        geom_sph = pin.GeometryObject(name, 0, sph, pin.SE3(np.eye(3), pos))
        self.vis_model.addGeometryObject(geom_sph)

    def _build_problem(self, q0, cyl1_center, p_ee_term):
        nv = self.rmodel.nv
        crad = 0.09
        cyl1_geom = fcl.Cylinder(crad, 5.0)
        geom_cyl1 = pin.GeometryObject(
            "osbt1", 0, cyl1_geom, pin.SE3(np.eye(3), cyl1_center)
        )
        geom_cyl1.meshColor[:] = (0.2, 1.0, 1.0, 0.4)
        self.vis_model.addGeometryObject(geom_cyl1)
        # cyl2_center = np.array([+0.35, -0.3, 0.0])
        # cyl2_geom = fcl.Cylinder(crad, 5.0)
        # geom_cyl2 = pin.GeometryObject(
        #     "osbt2", 0, cyl2_geom, pin.SE3(np.eye(3), cyl2_center)
        # )
        # geom_cyl2.meshColor[:] = (0.2, 1.0, 1.0, 0.4)
        # self.vis_model.addGeometryObject(geom_cyl2)

        rmodel = self.rmodel
        space = manifolds.MultibodyPhaseSpace(rmodel)
        ndx = space.ndx
        nu = nv

        v0 = np.zeros(nv)
        x0 = np.concatenate((q0, v0))

        self.coll_model.geometryObjects[0].geometry.computeLocalAABB()
        geom_names = (
            "ee_link_0",
            "forearm_link_0",
            "wrist_1_link_0",
            "upper_arm_link_0",
        )
        geom_ids = [self.coll_model.getGeometryId(name) for name in geom_names]

        self.add_objective_viz("ee_term", p_ee_term)
        ee_midway = np.array([0.8, 0.1, 0.4])
        self.add_objective_viz("ee_mid", ee_midway)

        ode = aligator.dynamics.MultibodyFreeFwdDynamics(space)

        self.dt = 0.01
        Tf = 2.4
        nsteps = int(Tf / self.dt)
        self.times = np.linspace(0.0, Tf, nsteps + 1)

        tau0 = pin.rnea(rmodel, self.rdata, q0, v0, v0)
        dyn_model = aligator.dynamics.IntegratorSemiImplEuler(ode, self.dt)

        us_init = [tau0] * nsteps
        xs_init = aligator.rollout(dyn_model, x0, us_init)

        ee_frame_id = rmodel.getFrameId("tool0")
        coll_frames = []
        coll_radii = []
        for gid in geom_ids:
            gobj: pin.GeometryObject = self.coll_model.geometryObjects[gid]
            fid = gobj.parentFrame
            jid = gobj.parentJoint
            coll_frames.append(fid)

            gobj.geometry.computeLocalAABB()
            radius = 0.1

            _collsph = pin.GeometryObject(
                f"{gobj.name}_sph", jid, fid, pin.SE3.Identity(), fcl.Sphere(radius)
            )
            _collsph.meshColor[:] = (0.8, 0.8, 0.0, 0.2)
            self.vis_model.addGeometryObject(_collsph)

            coll_radii.append(radius)

        stages = []

        for i in range(nsteps):
            rcost = aligator.CostStack(space, nu)
            Wx = 1e-2 * np.ones(ndx)
            Wx[nv:] = 1e-1
            rcost.addCost(
                "xreg",
                aligator.QuadraticStateCost(space, nu, x0, self.dt * np.diag(Wx)),
            )
            Wu = 1e-3 * np.eye(nu)
            rcost.addCost(
                "ureg", aligator.QuadraticControlCost(space, nu, self.dt * Wu)
            )
            stage = aligator.StageModel(cost=rcost, dynamics=dyn_model)

            if i == nsteps / 2:
                frame_res = aligator.FrameTranslationResidual(
                    ndx, nu, rmodel, ee_midway, ee_frame_id
                )
                stage.addConstraint(frame_res, constraints.EqualityConstraintSet())
            coldist1 = SphereCylinderCollisionDistance(
                rmodel,
                ndx,
                nu,
                cyl1_center[:2],
                cyl1_geom.radius,
                coll_radii,
                coll_frames,
            )
            stage.addConstraint(coldist1, constraints.NegativeOrthant())
            # coldist2 = SphereCylinderCollisionDistance(
            #     rmodel,
            #     ndx,
            #     nu,
            #     cyl2_center[:2],
            #     cyl2_geom.radius,
            #     coll_radii,
            #     coll_frames,
            # )
            # stage.addConstraint(coldist2, constraints.NegativeOrthant())
            # stage.addConstraint(
            #     aligator.ControlErrorResidual(ndx, nu),
            #     constraints.BoxConstraint(-rmodel.effortLimit, rmodel.effortLimit),
            # )
            stages.append(stage)

        frame_res = aligator.FrameTranslationResidual(
            ndx, nu, rmodel, p_ee_term, ee_frame_id
        )
        term_cost = aligator.CostStack(space, nu)
        Wxterm = 1e-4 * np.eye(ndx)
        term_cost.addCost("xreg", aligator.QuadraticStateCost(space, nu, x0, Wxterm))
        problem = aligator.TrajOptProblem(x0, stages=stages, term_cost=term_cost)
        problem.addTerminalConstraint(frame_res, constraints.EqualityConstraintSet())
        return problem, xs_init, us_init


if __name__ == "__main__":
    from pinocchio.visualize.meshcat_visualizer import MeshcatVisualizer
    from .solver_runner import ProxDdpRunner, IpoptRunner, AltroRunner

    args = Args().parse_args()
    example = UrSlalomExample()
    nq = example.rmodel.nq
    problem = example.problem
    xs_i = example.xs_init
    us_i = example.us_init
    dt = example.dt

    if args.viz:
        viz = MeshcatVisualizer(example.rmodel, example.coll_model, example.vis_model)
        viz.initViewer(open=True, loadModel=True)
        viz.display(example.robot.q0)

    tol = 1e-3
    mu_init = 20.0
    MAX_ITERS = 200
    match args.solver:
        case "ipopt":
            runner = IpoptRunner(
                {
                    "hessian_approximation": "limited-memory",
                    "print_level": 3,
                    "max_iters": MAX_ITERS,
                }
            )
            res = runner.solve(example, tol)
            print(res)
            xs_ = np.stack(runner.solver.xs)
            us_ = np.stack(runner.solver.us)
        case "ali":
            runner = ProxDdpRunner(
                {
                    "mu_init": mu_init,
                    "verbose": True,
                    # "ls_eta": 0.0,
                    "rollout_type": "linear",
                    "max_iters": MAX_ITERS,
                }
            )
            res = runner.solve(example, tol)
            print(res)
            aliresults = runner.solver.results
            print(aliresults)
            xs_ = np.stack(aliresults.xs)
            us_ = np.stack(aliresults.us)
        case "altro":
            runner = AltroRunner(
                {"mu_init": 1.0, "verbose": True, "max_iters": MAX_ITERS}
            )
            res = runner.solve(example, tol)
            altro_solve = runner.solver
            print(res)
            xs_ = np.stack(altro_solve.GetAllStates())
            us_ = np.stack(altro_solve.GetAllInputs())

    qs_ = xs_[:, :nq]

    if args.plot:
        import matplotlib.pyplot as plt
        from aligator.utils import plotting

        plotting.plot_controls_traj(times=example.times, us=us_, rmodel=example.rmodel)
        plt.show()

    if args.viz:
        import contextlib

        ctx = (
            viz.create_video_ctx("ur_slalom.mp4", fps=1 / dt)
            if args.record
            else contextlib.nullcontext()
        )

        with ctx:
            for i in range(3):
                input("[enter]")
                viz.play(qs_)
