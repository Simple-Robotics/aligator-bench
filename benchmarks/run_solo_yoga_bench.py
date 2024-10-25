import polars

from .solo import SoloYoga
from .solver_runner import ProxDdpRunner, IpoptRunner
from .bench_runner import SolverConfig, run_benchmark_configs


TOL = 1e-3
MAX_ITERS = 400

instances = [
    {"dip_angle": 40.0},
    {"dip_angle": 35.0},
    {"dip_angle": 30.0},
    {"dip_angle": 25.0},
    {"dip_angle": 20.0},
    {"dip_angle": 15.0},
]

SOLVERS: list[SolverConfig] = [
    (ProxDdpRunner, {"mu_init": 1e-4, "rollout_type": "linear"}),
    (IpoptRunner, {"print_level": 3}),
]

for _, settings in SOLVERS:
    settings["verbose"] = False
    settings["max_iters"] = MAX_ITERS


df = run_benchmark_configs(
    "solo_yoga", SoloYoga, TOL, instance_configs=instances, solver_configs=SOLVERS
)

with polars.Config(tbl_rows=-1):
    print(df)
