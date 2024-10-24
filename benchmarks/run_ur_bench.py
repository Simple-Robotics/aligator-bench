import numpy as np
import polars
from .ur5_problem import URProblem, generate_random_ee_target
from .solver_runner import AltroRunner, ProxDdpRunner, IpoptRunner
from .bench_runner import run_benchmark_configs, save_run

SEED = 42
np.random.seed(SEED)


rmodel = URProblem.robot.model


def run_with_vel(vel: bool, name):
    import aligator

    num_instances = 5
    instance_configs = []
    for i in range(num_instances):
        ee_target = generate_random_ee_target()
        instance_configs.append({"vel_constraint": vel, "ee_target": ee_target})

    TOL = 1e-4
    MAX_ITERS = 400

    mu_init = 1.0 if vel else 1.0
    default_start = True
    SOLVERS = [
        (AltroRunner, {"mu_init": 1.0, "tol_stationarity": 1e-4}),
        (
            ProxDdpRunner,
            {
                "mu_init": mu_init,
                "default_start": default_start,
                "rollout_type": aligator.ROLLOUT_LINEAR
                if vel
                else aligator.ROLLOUT_NONLINEAR,
            },
        ),
        (IpoptRunner, {"print_level": 2, "default_start": default_start}),
    ]

    for _, settings in SOLVERS:
        settings["verbose"] = False
        settings["max_iters"] = MAX_ITERS

    df = run_benchmark_configs(
        cls=URProblem,
        tol=TOL,
        instance_configs=instance_configs,
        solver_configs=SOLVERS,
    )

    with polars.Config(tbl_rows=-1, tbl_cols=10):
        print(df)

    save_run(df, name)


# run_with_vel(False, "ur5_reach")
run_with_vel(True, "ur5_reach_vel")
