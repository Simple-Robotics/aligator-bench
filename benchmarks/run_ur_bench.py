import numpy as np
import pinocchio as pin
from .ur5_problem import URProblem, generate_random_ee_target
from .solver_runner import AltroRunner, ProxDdpRunner, IpoptRunner
from .bench_runner import run_benchmark_configs

SEED = 42
np.random.seed(SEED)


rmodel = URProblem.robot.model


def run_with_vel(vel: bool, name):
    num_instances = 25
    q0_def = pin.neutral(rmodel)
    instance_configs = []
    for i in range(num_instances):
        ee_target = generate_random_ee_target()
        q0_gen = q0_def + np.random.randn(rmodel.nq)
        instance_configs.append(
            {"vel_constraint": vel, "ee_target": ee_target, "q0": q0_gen}
        )

    default_start = False
    MAX_MAX_ITERS = 400
    max_iter = 50

    # Sweep through values of max_iter
    while max_iter <= MAX_MAX_ITERS:
        SOLVERS = [
            (AltroRunner, {"mu_init": 1.0, "tol_stationarity": 1e-4}),
            (
                ProxDdpRunner,
                {
                    "mu_init": 1.0,
                    "default_start": default_start,
                    "rollout_type": "nonlinear",
                },
            ),
            (
                ProxDdpRunner,
                {
                    "mu_init": 1e-1,
                    "default_start": default_start,
                    "rollout_type": "nonlinear",
                },
            ),
            (
                ProxDdpRunner,
                {
                    "mu_init": 1e-2,
                    "default_start": default_start,
                    "rollout_type": "nonlinear",
                },
            ),
            (
                ProxDdpRunner,
                {
                    "mu_init": 1e-1,
                    "default_start": default_start,
                    "rollout_type": "linear",
                },
            ),
            (IpoptRunner, {"print_level": 2, "default_start": default_start}),
        ]

        for _, settings in SOLVERS:
            settings["verbose"] = False
            settings["max_iters"] = max_iter

        TOL = 1e-4
        run_benchmark_configs(
            bench_name=f"{name}_{max_iter}",
            cls=URProblem,
            tol=TOL,
            instance_configs=instance_configs,
            solver_configs=SOLVERS,
        )

        max_iter += 50


run_with_vel(False, "ur5_reach")
run_with_vel(True, "ur5_reach_vel")
