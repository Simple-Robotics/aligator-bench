import numpy as np
import pinocchio as pin
import polars
from .ur5_problem import URProblem, get_default_config_ee_pose
from .solver_runner import AltroRunner, ProxDdpRunner, IpoptRunner
from .bench_runner import run_benchmark_configs

SEED = 42
np.random.seed(42)


rmodel = URProblem.robot.model
default_ee_target = get_default_config_ee_pose(rmodel, URProblem.EE_NAME)

num_instances = 5
instance_configs = []
for i in range(num_instances):
    s = np.random.uniform(0.1, 0.95)
    th = np.random.uniform(0, 2 * np.pi)
    _zaxis = np.array([0.0, 0.0, 1.0])
    _R = pin.exp3(th * _zaxis)
    ee_target = s * _R @ default_ee_target
    config = {"vel_constraint": False, "ee_target": ee_target}
    instance_configs.append(config)

TOL = 1e-4
MAX_ITERS = 400

mu_init = 1.0
SOLVERS = [
    (AltroRunner, {"mu_init": mu_init}),
    (ProxDdpRunner, {"mu_init": mu_init}),
    (IpoptRunner, {"print_level": 2}),
]


for _, settings in SOLVERS:
    settings["verbose"] = False
    settings["max_iters"] = MAX_ITERS


data_ = run_benchmark_configs(
    cls=URProblem, tol=TOL, instance_configs=instance_configs, solver_configs=SOLVERS
)

df_ = polars.DataFrame(data_)
with polars.Config(tbl_rows=-1):
    print(df_)
