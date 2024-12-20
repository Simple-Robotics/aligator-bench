import numpy as np

from pathlib import Path
from .ur_slalom import UrSlalomExample, default_p_ee_term
from .solver_runner import AltroRunner, ProxDdpRunner, Status
from .bench_runner import run_benchmark_configs
from aligator import TrajOptProblem
from tap import Tap


class Args(Tap):
    seed: int = 1515
    pool_size: int = 3


args = Args().parse_args()

np.random.seed(args.seed)


def _run_random_init(run_args):
    tol, cls, config, (runner_cls, runner_settings) = run_args
    example: UrSlalomExample = cls(**config)
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


def main(args: Args, bench_name):
    import pickle

    print(args)

    default_start = False
    MAX_ITER = 200

    _proxddp_ls_etas = [0.0, 0.85]

    SOLVERS = [
        (AltroRunner, {"mu_init": 1.0, "tol_stationarity": 1e-4}),
    ]
    for avg_eta in _proxddp_ls_etas:
        SOLVERS.extend(
            [
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
                        "mu_init": 10.0,
                        "default_start": default_start,
                        "rollout_type": "nonlinear",
                        "ls_eta": avg_eta,
                    },
                ),
                (
                    ProxDdpRunner,
                    {
                        "mu_init": 10.0,
                        "default_start": default_start,
                        "rollout_type": "linear",
                        "ls_eta": avg_eta,
                    },
                ),
            ]
        )
    # SOLVERS.extend(
    #     [
    #         (
    #             IpoptRunner,
    #             {
    #                 "print_level": 2,
    #                 "default_start": default_start,
    #                 "hessian_approximation": "exact",
    #             },
    #         ),
    #         (
    #             IpoptRunner,
    #             {
    #                 "print_level": 2,
    #                 "default_start": default_start,
    #                 "hessian_approximation": "limited-memory",
    #             },
    #         ),
    #     ]
    # )

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
            cls=UrSlalomExample,
            tol=TOL,
            instance_configs=instance_configs,
            solver_configs=SOLVERS,
            pool_size=args.pool_size,
        )
    else:
        print(f"No problems found at {problems_path.absolute()}, generating some...")
        # Generate instances
        num_instances = 50
        instance_configs = []
        for jj in range(num_instances):
            p_ee_term = default_p_ee_term.copy()
            p_ee_term += 0.05 * np.random.randn(3)
            rmodel = UrSlalomExample.rmodel
            q0 = UrSlalomExample.robot.q0 + np.random.randn(rmodel.nq)
            config = {"q0": q0, "p_ee_term": p_ee_term}
            instance_configs.append(config)

        # save checkpoint
        with problems_path.open("wb") as f:
            pickle.dump(instance_configs, f)


main(args, "ur_slalom")
