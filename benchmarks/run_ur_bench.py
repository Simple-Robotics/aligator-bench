import numpy as np
import pinocchio as pin
import dataclasses
import polars
from .ur5_problem import URProblem, get_default_config_ee_pose
from .solver_runner import AltroRunner, ProxDdpRunner, IpoptRunner

SEED = 42
np.random.seed(42)


rmodel = URProblem.robot.model
default_ee_target = get_default_config_ee_pose(rmodel, URProblem.EE_NAME)

instance_configs = []

for i in range(4):
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


data_ = []


for i, config in enumerate(instance_configs):
    example = URProblem(**config)
    for runner_cls, settings in SOLVERS:
        runner = runner_cls(settings)
        entry = {"name": runner.name()}
        print(runner.name(), settings)
        print(example.name(), config)
        res = runner.solve(example, TOL)
        print(res)
        print("-------")
        entry.update(dataclasses.asdict(res))
        instance_name = example.name() + f"_{i}"
        entry["instance"] = instance_name
        data_.append(entry)


df_ = polars.DataFrame(data_)
with polars.Config(tbl_rows=-1):
    print(df_)
