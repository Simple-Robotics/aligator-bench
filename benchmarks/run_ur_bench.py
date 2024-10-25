import numpy as np
import polars
from .ur5_problem import URProblem, generate_random_ee_target
from .solver_runner import AltroRunner, ProxDdpRunner, IpoptRunner
from .bench_runner import run_benchmark_configs

SEED = 42
np.random.seed(SEED)


rmodel = URProblem.robot.model


def run_with_vel(vel: bool, name):
    num_instances = 10
    instance_configs = []
    for i in range(num_instances):
        ee_target = generate_random_ee_target()
        instance_configs.append({"vel_constraint": vel, "ee_target": ee_target})

    TOL = 1e-4
    MAX_ITERS = 400

    default_start = False
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
                "rollout_type": "linear",
            },
        ),
        (IpoptRunner, {"print_level": 2, "default_start": default_start}),
    ]

    for _, settings in SOLVERS:
        settings["verbose"] = False
        settings["max_iters"] = MAX_ITERS

    df = run_benchmark_configs(
        bench_name=name,
        cls=URProblem,
        tol=TOL,
        instance_configs=instance_configs,
        solver_configs=SOLVERS,
    )

    with polars.Config(tbl_rows=-1, tbl_cols=10):
        print(df)


run_with_vel(False, "ur5_reach")
# run_with_vel(True, "ur5_reach_vel")
