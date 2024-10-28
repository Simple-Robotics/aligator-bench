import numpy as np
import pinocchio as pin

from pathlib import Path
from .ur5_problem import URProblem, generate_random_ee_target
from .solver_runner import AltroRunner, ProxDdpRunner, IpoptRunner, Status
from .bench_runner import run_benchmark_configs
from aligator import TrajOptProblem

SEED = 42
np.random.seed(SEED)


def _run_random_init(args):
    tol, cls, config, (runner_cls, runner_settings) = args
    example: URProblem = cls(**config)
    rmodel = example.rmodel
    p: TrajOptProblem = example.problem
    nsteps = p.num_steps
    nq = rmodel.nq
    nv = rmodel.nv
    nx = nq + nv
    std = 3.0
    xs_init = [std * np.random.randn(nx) for _ in range(nsteps + 1)]
    us_init = [std * np.random.randn(nv) for _ in range(nsteps)]
    runner = runner_cls({**runner_settings, "warm_start": (xs_init, us_init)})
    res = runner.solve(example, tol)
    return res.status == Status.CONVERGED.name


def check_if_problem_feasible(tol, cls, config, runner_configs: list):
    """Check if any solver-config pair solves the example given a random warm-start."""
    from multiprocessing import Pool

    args = [(tol, cls, config, sc) for sc in runner_configs]
    print(f"Checking if problem feasible with {len(args)} configs.")
    with Pool(2) as pool:
        for res in pool.imap_unordered(_run_random_init, args, chunksize=1):
            if res:
                pool.terminate()
                print("PROBLEM FEASIBLE")
                return True
    return False


def run_with_vel(vel: bool, bench_name):
    import pickle

    rmodel = URProblem.rmodel
    default_start = False
    MAX_ITER = 400

    _proxddp_ls_etas = [0.2, 0.85]

    SOLVERS = [
        (AltroRunner, {"mu_init": 1.0, "tol_stationarity": 1e-4}),
        (
            IpoptRunner,
            {
                "print_level": 2,
                "default_start": default_start,
                "hessian_approximation": "exact",
            },
        ),
        (
            IpoptRunner,
            {
                "print_level": 2,
                "default_start": default_start,
                "hessian_approximation": "limited-memory",
            },
        ),
    ]
    for avg_eta in _proxddp_ls_etas:
        SOLVERS += [
            (
                ProxDdpRunner,
                {
                    "mu_init": 1.0,
                    "default_start": default_start,
                    "rollout_type": "nonlinear",
                    "ls_eta": avg_eta,
                },
            ),
            (
                ProxDdpRunner,
                {
                    "mu_init": 1e-1,
                    "default_start": default_start,
                    "rollout_type": "linear",
                    "ls_eta": avg_eta,
                },
            ),
        ]

    for _, settings in SOLVERS:
        settings["verbose"] = False
        settings["max_iters"] = MAX_ITER

    TOL = 1e-4

    problems_path = Path(f"{bench_name}_problems.pkl")
    if problems_path.exists():
        print(f"Loading pre-selected problems from {problems_path.absolute()}...")
        with problems_path.open("rb") as f:
            instance_configs = pickle.load(f)
        assert len(instance_configs) >= 1
        assert isinstance(instance_configs[0], dict)
        print(f"Loaded {len(instance_configs)} problem configurations.")

        run_benchmark_configs(
            bench_name=bench_name,
            cls=URProblem,
            tol=TOL,
            instance_configs=instance_configs,
            solver_configs=SOLVERS,
        )
    else:
        print(f"No problems found at {problems_path.absolute()}, generating some...")
        # Generate instances
        num_instances = 40
        q0_def = pin.neutral(rmodel)
        instance_configs = []
        _jj = 0
        while _jj < num_instances:
            ee_target = generate_random_ee_target()
            q0_gen = q0_def + np.random.randn(rmodel.nq)
            config = {"vel_constraint": vel, "ee_target": ee_target, "q0": q0_gen}
            if not vel or check_if_problem_feasible(
                tol=TOL, cls=URProblem, config=config, runner_configs=SOLVERS
            ):
                instance_configs.append(config)
                _jj += 1

        # save checkpoint
        with problems_path.open("wb") as f:
            pickle.dump(instance_configs, f)


# run_with_vel(False, "ur5_reach")
run_with_vel(True, "ur5_reach_vel")
