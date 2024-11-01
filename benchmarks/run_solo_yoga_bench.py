import numpy as np

from .solo import SoloYoga
from .solver_runner import ProxDdpRunner, IpoptRunner
from .bench_runner import SolverConfig, run_benchmark_configs
from tap import Tap


class Args(Tap):
    seed: int = 1515
    pool_size: int


TOL = 1e-3
num_instances = 50
args = Args().parse_args()
np.random.seed(args.seed)

MAX_ITERS = 500
instances = []
for i in range(num_instances):
    instances.append(
        {
            "dip_angle": np.random.uniform(10.0, 50.0),
            "twist_angle": np.random.uniform(-20.0, 20.0),
        }
    )

SOLVERS: list[SolverConfig] = [
    (
        IpoptRunner,
        {
            "print_level": 2,
            "hessian_approximation": "exact",
        },
    ),
    (
        IpoptRunner,
        {
            "print_level": 2,
            "hessian_approximation": "limited-memory",
        },
    ),
]
ls_etas_ = [0.0, 0.2, 0.85]
for ls_eta in ls_etas_:
    SOLVERS += [
        (
            ProxDdpRunner,
            {"mu_init": 1e-2, "rollout_type": "linear", "ls_eta": ls_eta},
        ),
        (
            ProxDdpRunner,
            {"mu_init": 1e-4, "rollout_type": "linear", "ls_eta": ls_eta},
        ),
    ]

for _, settings in SOLVERS:
    settings["verbose"] = False
    settings["max_iters"] = MAX_ITERS

run_benchmark_configs(
    bench_name="solo_yoga",
    cls=SoloYoga,
    tol=TOL,
    instance_configs=instances,
    solver_configs=SOLVERS,
    pool_size=args.pool_size,
)
