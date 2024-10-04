import numpy as np
from .ur5_problem import URProblem
from .solver_runner import AltroRunner, ProxDdpRunner

SEED = 42
np.random.seed(42)


rmodel = URProblem.robot.model
default_ee_target = URProblem.get_default_config_ee_pose(rmodel)

instance_configs = []

for i in range(4):
    s = np.random.uniform(0.1, 0.95)
    ee_target = s * default_ee_target
    config = {"vel_constraint": False, "ee_target": ee_target}
    instance_configs.append(config)


SOLVERS = [(AltroRunner, {}), (ProxDdpRunner, {"mu_init": 1.0})]

TOL = 1e-5

for _, settings in SOLVERS:
    settings["verbose"] = True


for runner_cls, settings in SOLVERS:
    for config in instance_configs:
        print(config)
        example = URProblem(**config)
        runner = runner_cls(settings)
        res = runner.solve(example, TOL)
        print(res)
